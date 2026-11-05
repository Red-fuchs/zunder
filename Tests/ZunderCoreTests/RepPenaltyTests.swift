import MLX
import XCTest
@testable import ZunderCore

final class RepPenaltyTests: XCTestCase {
  func testSignAware() {
    let l = MLXArray([2.0, -2.0, 3.0, 4.0] as [Float])
    let r = repPenalty(l, MLXArray([0, 1, 1] as [Int32]), 2).asArray(Float.self)
    XCTAssertEqual(r, [1.0, -4.0, 3.0, 4.0])
  }
}
