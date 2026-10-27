import XCTest
import MLX
import MLXLMCommon
@testable import ZunderCore

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

final class ParityTests: XCTestCase {
  func testTinyFlintLogits() async throws {
    let ref = try loadArrays(url: root.appendingPathComponent("ref/out/tiny.safetensors"))
    let ctx = try await loadContext(root.appendingPathComponent("fixtures/tiny-flint"))
    let cache = try ctx.model.newCache(parameters: nil)
    let ids = ref["ids"]!.asArray(Int32.self), nxt = ref["nxt"]!.asArray(Int32.self)
    var lg = ctx.model(MLXArray(ids)[.newAxis], cache: cache).asType(.float32)
    var worst: Float = 0
    func cmp(_ name: String, _ a: MLXArray, _ b: MLXArray) {
      let d = abs(a - b).max().item(Float.self)
      let am = argMax(a, axis: -1), bm = argMax(b, axis: -1)
      let eq = (am .== bm).all().item(Bool.self)
      print("parity \(name): maxabs=\(d) argmax_equal=\(eq)")
      XCTAssertTrue(eq, name); worst = max(worst, d)
    }
    cmp("prefill_head", lg[0, 0 ..< 4], ref["prefill_head"]!)
    cmp("prefill_last", lg[0, -1], ref["prefill_last"]!)
    for (i, t) in nxt.enumerated() {
      lg = ctx.model(MLXArray([t])[.newAxis], cache: cache).asType(.float32)
      cmp("step\(i)", lg[0, -1], ref["step\(i)"]!)
    }
    print("parity worst maxabs=\(worst)")
    XCTAssertLessThan(worst, 0.05)
  }
}
