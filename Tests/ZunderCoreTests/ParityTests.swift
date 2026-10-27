import XCTest
import MLX
import MLXLMCommon
@testable import ZunderCore

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

final class ParityTests: XCTestCase {
  var worst: Float = 0
  func cmp(_ name: String, _ a: MLXArray, _ b: MLXArray, tol: Float = 0.05) {
    let d = abs(a - b).max().item(Float.self)
    // random weights give near-flat logits: accept a different argmax only if it is a tie within 0.02 in the reference
    let ia = argMax(a, axis: -1)
    let eq = (argMax(b, axis: -1) .== ia).all().item(Bool.self) || (b.max() - b[ia]).item(Float.self) < 0.02
    print("parity \(name): maxabs=\(d) argmax_equal=\(eq)")
    XCTAssertTrue(eq, name); XCTAssertLessThan(d, tol, name); worst = max(worst, d)
  }

  func run(_ model: String, _ refFile: String) async throws {
    worst = 0
    let ref = try loadArrays(url: root.appendingPathComponent("ref/out/\(refFile).safetensors"))
    let ctx = try await loadContext(root.appendingPathComponent("fixtures/\(model)"))
    let m = ctx.model
    let ids = ref["ids"]!.asArray(Int32.self), nxt = ref["nxt"]!.asArray(Int32.self), wtok = ref["wtok"]!.asArray(Int32.self)
    func step(_ t: Int32, _ c: [KVCache]) -> MLXArray { m(MLXArray([t])[.newAxis], cache: c).asType(.float32)[0, -1] }

    // A: single-shot prefill (700 > window 513) + decode
    var cache = try m.newCache(parameters: nil)
    var lg = m(MLXArray(ids)[.newAxis], cache: cache).asType(.float32)
    cmp("\(model) prefill_head", lg[0, 0 ..< 4], ref["prefill_head"]!)
    cmp("\(model) prefill_last", lg[0, -1], ref["prefill_last"]!)
    for (i, t) in nxt.enumerated() { cmp("\(model) step\(i)", step(t, cache), ref["step\(i)"]!) }

    // B: chunked prefill 512 + 188 (what Engine does for long prompts), then decode
    cache = try m.newCache(parameters: nil)
    _ = m(MLXArray(Array(ids[..<512]))[.newAxis], cache: cache)
    lg = m(MLXArray(Array(ids[512...]))[.newAxis], cache: cache).asType(.float32)
    cmp("\(model) chunk_last", lg[0, -1], ref["chunk_last"]!)
    for (i, t) in nxt.enumerated() { cmp("\(model) cstep\(i)", step(t, cache), ref["cstep\(i)"]!) }

    // B2: Engine's own path (prepare() with stepSize 512, balanced chunks) vs the single-shot reference
    cache = try m.newCache(parameters: nil)
    let inp = LMInput(tokens: MLXArray(ids))
    switch try m.prepare(inp, cache: cache, state: nil, prefill: .init(stepSize: 512)) {
    case .tokens(let t): lg = m(t[text: .newAxis], cache: cache, state: nil).logits.asType(.float32)
    case .logits(let o): lg = o.logits.asType(.float32)
    }
    cmp("\(model) prepare_last", lg[0, -1], ref["prefill_last"]!)

    // C: teacher-forced decode 620 tokens, far past the rotating-cache wrap (700 + 620 > 2 * 513)
    cache = try m.newCache(parameters: nil)
    _ = m(MLXArray(ids)[.newAxis], cache: cache)
    for (i, t) in wtok.enumerated() {
      let l = step(t, cache)
      if i % 100 == 99 || i == wtok.count - 1 { cmp("\(model) wrap w\(i)", l, ref["w\(i)"]!) }
    }
    print("parity \(model) worst maxabs=\(worst)")
  }

  func testTinyBF16() async throws { try await run("tiny-flint", "tiny") }
  /// 4-bit g32 with 8-bit embeddings and NO per-layer entries in config.json: needs the inferred per-tensor bits
  func testTinyMixedBits() async throws {
    let dir = root.appendingPathComponent("fixtures/tiny-flint-q")
    let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))
    let inf = try XCTUnwrap(try inferQuantization(dir: dir, base: base.perLayerQuantization!))
    XCTAssertEqual(inf.quantization(layer: "model.embed_tokens")?.bits, 8)
    XCTAssertEqual(inf.quantization(layer: "lm_head")?.bits, 4)
    XCTAssertEqual(inf.quantization(layer: "model.layers.2.mlp.switch_mlp.up_proj")?.bits, 4)
    try await run("tiny-flint-q", "tinyq")
  }
}

final class QuantInferTests: XCTestCase {
  func testFlintInt4Layout() {
    // shapes as in flint-int4: uint8-packed weights, global 4 bit g64, 8-bit embedding
    let h: [String: TensorInfo] = [
      "model.embed_tokens.weight": .init(dtype: "U8", shape: [150144, 2048 * 8 / 8]),   // 8 bit: H/4 u32 = H bytes
      "model.embed_tokens.scales": .init(dtype: "BF16", shape: [150144, 32]),
      "lm_head.weight": .init(dtype: "U8", shape: [150144, 1024]),                      // 4 bit: H/8 u32 = H/2 bytes
      "lm_head.scales": .init(dtype: "BF16", shape: [150144, 32]),
      "model.layers.3.mlp.experts.7.up_proj.weight": .init(dtype: "U32", shape: [512, 256]),
      "model.layers.3.mlp.experts.7.up_proj.scales": .init(dtype: "BF16", shape: [512, 32]),
      "model.layers.3.mlp.gate.weight": .init(dtype: "BF16", shape: [256, 2048]),
    ]
    let base = BaseConfiguration.PerLayerQuantization(quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [:])
    let r = inferQuantization(h, base: base)
    XCTAssertEqual(r?.quantization(layer: "model.embed_tokens")?.bits, 8)
    XCTAssertEqual(r?.quantization(layer: "lm_head")?.bits, 4)
    XCTAssertEqual(r?.quantization(layer: "model.layers.3.mlp.switch_mlp.up_proj")?.bits, 4)
    XCTAssertEqual(r?.quantization(layer: "model.embed_tokens")?.groupSize, 64)
  }
  func testUniformReturnsNil() {
    let h: [String: TensorInfo] = ["a.weight": .init(dtype: "U32", shape: [8, 256]), "a.scales": .init(dtype: "BF16", shape: [8, 32])]
    XCTAssertNil(inferQuantization(h, base: .init(quantization: .init(groupSize: 64, bits: 4), perLayerQuantization: [:])))
  }
}
