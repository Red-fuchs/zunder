// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/flint.py (Apache-2.0, Copyright Apple Inc.)
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct FlintConfiguration: Codable, Sendable {
  var modelType: String
  var hiddenSize: Int
  var layers: Int
  var intermediateSize: Int
  var heads: Int
  var kvHeads: Int
  var headDim: Int
  var experts: Int
  var expertsPerTok: Int
  var moeIntermediateSize: Int
  var sharedIntermediateSize: Int
  var mlpOnlyLayers: [Int]
  var layerTypes: [String]
  var slidingWindow: Int
  var eps: Float
  var vocabSize: Int
  var ropeTheta: Float = 10000
  var tieWordEmbeddings = false

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type", hiddenSize = "hidden_size", layers = "num_hidden_layers"
    case intermediateSize = "intermediate_size", heads = "num_attention_heads", kvHeads = "num_key_value_heads"
    case headDim = "head_dim", experts = "num_experts", expertsPerTok = "num_experts_per_tok"
    case moeIntermediateSize = "moe_intermediate_size", sharedIntermediateSize = "shared_expert_intermediate_size"
    case mlpOnlyLayers = "mlp_only_layers", layerTypes = "layer_types", slidingWindow = "sliding_window"
    case eps = "rms_norm_eps", vocabSize = "vocab_size", ropeTheta = "rope_theta"
    case ropeParameters = "rope_parameters", tieWordEmbeddings = "tie_word_embeddings"
  }

  public init(from d: Decoder) throws {
    let c = try d.container(keyedBy: CodingKeys.self)
    modelType = try c.decode(String.self, forKey: .modelType)
    hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
    layers = try c.decode(Int.self, forKey: .layers)
    intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
    heads = try c.decode(Int.self, forKey: .heads)
    kvHeads = try c.decode(Int.self, forKey: .kvHeads)
    headDim = try c.decode(Int.self, forKey: .headDim)
    experts = try c.decode(Int.self, forKey: .experts)
    expertsPerTok = try c.decode(Int.self, forKey: .expertsPerTok)
    moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
    sharedIntermediateSize = try c.decode(Int.self, forKey: .sharedIntermediateSize)
    mlpOnlyLayers = try c.decode([Int].self, forKey: .mlpOnlyLayers)
    layerTypes = try c.decode([String].self, forKey: .layerTypes)
    slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
    eps = try c.decode(Float.self, forKey: .eps)
    vocabSize = try c.decode(Int.self, forKey: .vocabSize)
    ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
    if let rp = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters), let t = rp["rope_theta"]?.asFloat() { ropeTheta = t }
    tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
  }
  public func encode(to e: Encoder) throws {}
}

class FlintAttention: Module {
  let heads: Int, kvHeads: Int, scale: Float
  @ModuleInfo(key: "q_proj") var wq: Linear
  @ModuleInfo(key: "k_proj") var wk: Linear
  @ModuleInfo(key: "v_proj") var wv: Linear
  @ModuleInfo(key: "o_proj") var wo: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
  @ParameterInfo(key: "softmax_offset") var sinks: MLXArray
  let rope: RoPE?

  init(_ a: FlintConfiguration, layer: Int) {
    heads = a.heads; kvHeads = a.kvHeads; scale = pow(Float(a.headDim), -0.5)
    _wq.wrappedValue = Linear(a.hiddenSize, heads * a.headDim, bias: false)
    _wk.wrappedValue = Linear(a.hiddenSize, kvHeads * a.headDim, bias: false)
    _wv.wrappedValue = Linear(a.hiddenSize, kvHeads * a.headDim, bias: false)
    _wo.wrappedValue = Linear(heads * a.headDim, a.hiddenSize, bias: false)
    _qNorm.wrappedValue = RMSNorm(dimensions: a.headDim, eps: a.eps)
    _kNorm.wrappedValue = RMSNorm(dimensions: a.headDim, eps: a.eps)
    _sinks.wrappedValue = zeros([heads])
    rope = a.layerTypes[layer] == "sliding_attention" ? RoPE(dimensions: a.headDim, traditional: false, base: a.ropeTheta) : nil
  }

  func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))
    var q = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
    var k = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
    var v = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
    if let rope {
      let off = cache?.ropeOffset
      q = applyRotaryPosition(rope, to: q, offset: off)
      k = applyRotaryPosition(rope, to: k, offset: off)
    }
    if let cache { (k, v) = cache.update(keys: k, values: v) }
    // attentionWithCacheUpdate has no sinks parameter -> call SDPA directly
    let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask, sinks: sinks)
    return wo(o.transposed(0, 2, 1, 3).reshaped(B, L, -1))
  }
}

class FlintMLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear
  init(_ dim: Int, _ hidden: Int) {
    _gate.wrappedValue = Linear(dim, hidden, bias: false)
    _down.wrappedValue = Linear(hidden, dim, bias: false)
    _up.wrappedValue = Linear(dim, hidden, bias: false)
  }
  func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

class FlintMoE: Module, UnaryLayer {
  let k: Int
  @ModuleInfo(key: "gate") var gate: Linear
  @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
  @ModuleInfo(key: "shared_expert") var shared: FlintMLP
  init(_ a: FlintConfiguration) {
    k = a.expertsPerTok
    _gate.wrappedValue = Linear(a.hiddenSize, a.experts, bias: false)
    _switchMLP.wrappedValue = SwitchGLU(inputDims: a.hiddenSize, hiddenDims: a.moeIntermediateSize, numExperts: a.experts)
    _shared.wrappedValue = FlintMLP(a.hiddenSize, a.sharedIntermediateSize)
  }
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // pure-MLX routing branch of expert_routing: sigmoid, top-k, renormalise
    let scores = sigmoid(gate(x.asType(.float32)))
    let inds = stopGradient(argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k])
    var w = takeAlong(scores, inds, axis: -1)
    w = w / w.sum(axis: -1, keepDims: true)
    let y = switchMLP(x, inds)
    return weightedExpertSum(y, w).asType(y.dtype) + shared(x)
  }
}

class FlintLayer: Module {
  let sliding: Bool
  @ModuleInfo(key: "self_attn") var attn: FlintAttention
  let mlp: UnaryLayer
  @ModuleInfo(key: "input_layernorm") var inNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
  init(_ a: FlintConfiguration, layer: Int) {
    sliding = a.layerTypes[layer] == "sliding_attention"
    _attn.wrappedValue = FlintAttention(a, layer: layer)
    mlp = a.mlpOnlyLayers.contains(layer) ? FlintMLP(a.hiddenSize, a.intermediateSize) : FlintMoE(a)
    _inNorm.wrappedValue = RMSNorm(dimensions: a.hiddenSize, eps: a.eps)
    _postNorm.wrappedValue = RMSNorm(dimensions: a.hiddenSize, eps: a.eps)
  }
  func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
    let h = x + attn(inNorm(x), mask: mask, cache: cache)
    return h + mlp(postNorm(h))
  }
}

public class FlintInner: Module {
  @ModuleInfo(key: "embed_tokens") var embed: Embedding
  let layers: [FlintLayer]
  let norm: RMSNorm
  let window: Int, fa: Int?, swa: Int?
  init(_ a: FlintConfiguration) {
    _embed.wrappedValue = Embedding(embeddingCount: a.vocabSize, dimensions: a.hiddenSize)
    layers = (0 ..< a.layers).map { FlintLayer(a, layer: $0) }
    norm = RMSNorm(dimensions: a.hiddenSize, eps: a.eps)
    window = a.slidingWindow
    fa = a.layerTypes.firstIndex(of: "full_attention")
    swa = a.layerTypes.firstIndex(of: "sliding_attention")
  }
  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var h = embed(inputs)
    let n = h.dim(1)
    let faMask = fa.map { makeAttentionMask(n: n, cache: cache?[$0]) }
    let swaMask = swa.map { makeAttentionMask(n: n, cache: cache?[$0], windowSize: window) }
    for (i, l) in layers.enumerated() { h = l(h, mask: (l.sliding ? swaMask : faMask)!, cache: cache?[i]) }
    return norm(h)
  }
}

public class FlintModel: Module, LLMModel {
  public let vocabularySize: Int
  let cfg: FlintConfiguration
  let model: FlintInner
  @ModuleInfo(key: "lm_head") var lmHead: Linear?

  public init(_ a: FlintConfiguration) {
    cfg = a; vocabularySize = a.vocabSize
    model = FlintInner(a)
    if !a.tieWordEmbeddings { _lmHead.wrappedValue = Linear(a.hiddenSize, a.vocabSize, bias: false) }
  }

  public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    let h = model(inputs, cache: cache)
    return lmHead?(h) ?? model.embed.asLinear(h)
  }

  public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var w = weights
    if cfg.tieWordEmbeddings { w["lm_head.weight"] = nil }
    // flint-int4 stores packed weights as uint8
    for (k, v) in weights where v.dtype == .uint8 && k.hasSuffix(".weight") {
      if w[String(k.dropLast(7)) + ".scales"] != nil { w[k] = v.view(dtype: .uint32) }
    }
    for l in 0 ..< cfg.layers where !cfg.mlpOnlyLayers.contains(l) {
      let p = "model.layers.\(l).mlp"
      for n in ["up_proj", "down_proj", "gate_proj"] where w["\(p).experts.0.\(n).weight"] != nil {
        // quantised checkpoints also carry .scales/.biases per expert
        for s in ["weight", "scales", "biases"] where w["\(p).experts.0.\(n).\(s)"] != nil {
          w["\(p).switch_mlp.\(n).\(s)"] = MLX.stacked((0 ..< cfg.experts).map { w.removeValue(forKey: "\(p).experts.\($0).\(n).\(s)")! })
        }
      }
    }
    return w
  }

  public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
    try model.layers.map { try makeHybridAttentionKVCache(parameters: parameters, slidingWindow: cfg.slidingWindow, usesSlidingWindow: $0.sliding) }
  }

  // gate stays unquantised (quant_predicate in flint.py)
  public var loraLayers: [Module] { model.layers }
}

extension FlintModel {
  public var toolCallFormat: ToolCallFormat? { .glm4 }
  public var reasoningConfig: ReasoningConfig? { nil }
}

public func registerFlint() async {
  await LLMTypeRegistry.shared.registerModelType("flint") { data in
    FlintModel(try JSONDecoder.json5().decode(FlintConfiguration.self, from: data))
  }
}
