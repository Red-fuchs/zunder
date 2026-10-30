import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public struct ZError: Error, CustomStringConvertible { public let description: String; public init(_ s: String) { description = s } }

struct TokBridge: MLXLMCommon.Tokenizer {
  let u: any Tokenizers.Tokenizer
  func encode(text: String, addSpecialTokens: Bool) -> [Int] { u.encode(text: text, addSpecialTokens: addSpecialTokens) }
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { u.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens) }
  func convertTokenToId(_ t: String) -> Int? { u.convertTokenToId(t) }
  func convertIdToToken(_ i: Int) -> String? { u.convertIdToToken(i) }
  var bosToken: String? { u.bosToken }
  var eosToken: String? { u.eosToken }
  var unknownToken: String? { u.unknownToken }
  func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
    do { return try u.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext) }
    catch Tokenizers.TokenizerError.missingChatTemplate { throw MLXLMCommon.TokenizerError.missingChatTemplate }
  }
}

struct TokLoader: TokenizerLoader {
  func load(from d: URL) async throws -> any MLXLMCommon.Tokenizer { TokBridge(u: try await AutoTokenizer.from(modelFolder: d)) }
}

/// Local dir, or `org/name` resolved in the HF cache (never downloads).
public func resolveModel(_ spec: String) throws -> URL {
  let fm = FileManager.default
  var isDir: ObjCBool = false
  let p = (spec as NSString).expandingTildeInPath
  if fm.fileExists(atPath: p, isDirectory: &isDir), isDir.boolValue { return URL(fileURLWithPath: p) }
  let hub = ProcessInfo.processInfo.environment["HF_HOME"].map { $0 + "/hub" } ?? NSHomeDirectory() + "/.cache/huggingface/hub"
  let snaps = URL(fileURLWithPath: hub + "/models--" + spec.replacingOccurrences(of: "/", with: "--") + "/snapshots")
  let dirs = ((try? fm.contentsOfDirectory(at: snaps, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
    .filter { fm.fileExists(atPath: $0.appendingPathComponent("config.json").path) }
    .sorted { ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) > ((try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
  guard let d = dirs.first else { throw ZError("model not found (no download by design): \(spec)") }
  return d
}

public func loadContext(_ dir: URL, dtype: ActDtype = resolveDtype(nil)) async throws -> ModelContext {
  let c = try await loadContextRaw(dir)
  castParameters(c.model, to: dtype)
  if let m = c.model as? FlintModel { 
    let f16 = dtype == .fp16
    m.model.res32 = f16 && ProcessInfo.processInfo.environment["ZUNDER_RES32"] == "1"   // fp32 residual stream: off, residual max ~60 (not the overflow source)
    for l in m.model.layers { (l.mlp as? FlintMoE)?.safe = f16 && ProcessInfo.processInfo.environment["ZUNDER_SAFEMOE"] != "0" }
  }
  return c
}

func loadContextRaw(_ dir: URL) async throws -> ModelContext {
  await registerFlint()
  // mlx-swift-lm applies one global (bits, group) to every module with .scales; mixed checkpoints (flint-int4: 8-bit embeddings) need per-tensor bits
  let data = try Data(contentsOf: dir.appendingPathComponent("config.json"))
  let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
  guard let pl = base.perLayerQuantization, let inferred = try inferQuantization(dir: dir, base: pl) else {
    return try await LLMModelFactory.shared.load(from: dir, using: TokLoader())
  }
  let model = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: base.modelType)
  try await loadWeights(modelDirectory: dir, model: model, perLayerQuantization: inferred)
  return ModelContext(configuration: ModelConfiguration(directory: dir), model: model, processor: NoProcessor(), tokenizer: try await loadTokenizer(dir))
}

public func loadTokenizer(_ dir: URL) async throws -> any MLXLMCommon.Tokenizer { try await TokLoader().load(from: dir) }
