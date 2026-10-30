import CLLG
import Foundation

/// Per-token constraint behind `Gate`: xgrammar (`GrammarConstraint`) or llguidance (`LlgCon`).
protocol Constraint: AnyObject {
  /// allowed-token bitmask (bit i = token i, 32-bit words); nil = unrestricted
  func mask() throws -> [Int32]?
  func commit(_ id: Int) throws -> Int
  func rollback(_ n: Int) throws
}

public final class LlgTok {
  let p: OpaquePointer
  init(json: String, vocab: Int, eos: Int, extra: [Int]) throws {
    var ex = extra.filter { $0 != eos }.map { UInt32($0) }
    var i = LlgTokenizerInitV2()
    i.struct_size = MemoryLayout<LlgTokenizerInitV2>.size
    i.vocab_size = UInt32(vocab); i.tok_eos = UInt32(eos)
    i.use_approximate_greedy_tokenize_fn = true
    i.tok_eos_extra_count = UInt32(ex.count)
    var err = [CChar](repeating: 0, count: 512)
    let t = json.withCString { j -> OpaquePointer? in
      i.tokenizer_json = j
      return ex.withUnsafeMutableBufferPointer { b in i.tok_eos_extra = UnsafePointer(b.baseAddress); return llg_new_tokenizer_v2(&i, &err, err.count) }
    }
    guard let t else { throw ZError("llguidance tokenizer: " + String(cString: err)) }
    p = t
  }
  deinit { llg_free_tokenizer(p) }
}

final class LlgCon: Constraint {
  let m: OpaquePointer, words: Int
  init(tok: LlgTok, schema: String) throws {
    var i = LlgConstraintInit()
    llg_constraint_init_set_defaults(&i, tok.p)
    i.log_stderr_level = 0
    m = llg_new_matcher(&i, "json_schema", schema)
    if llg_matcher_is_error(m) { let e = String(cString: llg_matcher_get_error(m)); llg_free_matcher(m); throw ZError("llguidance schema: \(e)") }
    words = llg_matcher_get_mask_byte_size(m) / 4
  }
  deinit { llg_free_matcher(m) }
  func fail(_ w: String) -> ZError { ZError("llguidance \(w): " + (llg_matcher_get_error(m).map { String(cString: $0) } ?? "?")) }
  func mask() throws -> [Int32]? {
    guard llg_matcher_compute_mask(m) == 0, let p = llg_matcher_get_mask(m) else { throw fail("mask") }
    return Array(UnsafeBufferPointer(start: UnsafeRawPointer(p).assumingMemoryBound(to: Int32.self), count: words))
  }
  func commit(_ id: Int) throws -> Int { guard llg_matcher_consume_token(m, UInt32(id)) == 0 else { throw fail("commit") }; return 1 }
  func rollback(_ n: Int) throws { guard llg_matcher_rollback(m, n) == 0 else { throw fail("rollback") } }
}
