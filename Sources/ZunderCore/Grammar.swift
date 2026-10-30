import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon

/// xgrammar vocab for one tokenizer + compiled-constraint cache (matchers are reused via rollback).
public final class GrammarKit {
  public let gtok: GrammarTokenizer
  public let eos: Int
  public let stop: Set<Int>
  var cache: [String: any Constraint] = [:]
  var llg: LlgTok?
  let dir: URL?, logitDim: Int?
  /// JSON-schema backend: llguidance (default) or xgrammar (ZUNDER_SCHEMA_BACKEND=xgrammar)
  let useLlg = ProcessInfo.processInfo.environment["ZUNDER_SCHEMA_BACKEND"] != "xgrammar"

  public init(tok: any MLXLMCommon.Tokenizer, eos: Int, stop: Set<Int>, dir: URL? = nil, logitDim: Int? = nil) throws {
    self.dir = dir; self.logitDim = logitDim
    let v = TokenizerVocabExtractor.extractForGrammar(from: tok)
    gtok = try GrammarTokenizer(vocab: v.vocab, vocabType: v.vocabType, eosTokenId: Int32(eos))
    self.eos = eos; self.stop = stop.union([eos])
  }

  public var vocab: Int { gtok.vocabSize }

  /// Fresh-state matcher for `tag`; give it back with `release` so the compile is reused.
  func take(_ tag: String, schema: Bool = false, lang: String? = nil) throws -> (any Constraint, key: String, compile: Double) {
    let k = (lang.map { "G\($0)" } ?? (schema ? (useLlg ? "L" : "S") : "T")) + tag
    if let c = cache.removeValue(forKey: k) { return (c, k, 0) }
    let t = Date()
    let c: any Constraint
    if lang != nil || (schema && useLlg) {
      if llg == nil {
        guard let d = dir, let j = try? String(contentsOf: d.appendingPathComponent("tokenizer.json"), encoding: .utf8) else { throw ZError("llguidance needs tokenizer.json") }
        llg = try LlgTok(json: j, vocab: max(vocab, logitDim ?? 0), eos: eos, extra: Array(stop))
      }
      c = try LlgCon(tok: llg!, kind: lang ?? "json_schema", schema: tag)
    } else if schema { c = try GrammarConstraint(tokenizer: gtok, jsonSchema: tag) }
    else { c = try GrammarConstraint(tokenizer: gtok, structuralTag: tag) }
    return (c, k, Date().timeIntervalSince(t))
  }
  func release(_ c: any Constraint, key: String, committed: Int) {
    if committed > 0 { do { try c.rollback(committed) } catch { FileHandle.standardError.write(Data("grammar rollback failed (matcher dropped): \(error)\n".utf8)); return } }
    cache[key] = c
  }
  public func clear() { cache = [:] }
}

/// Per-request mask state: wraps a matcher, lets every stop token through when the grammar allows EOS.
public final class Gate {
  let c: any Constraint, kit: GrammarKit, key: String
  /// `after`: free generation until this token is committed, then the grammar starts (scope after_think)
  var after: Int?
  var ended = false
  public private(set) var committed = 0
  public var maskS = 0.0
  var shifts = MLXArray(Array(0 ..< 32).map { Int32($0) })
  init(_ c: any Constraint, kit: GrammarKit, key: String) { self.c = c; self.kit = kit; self.key = key }

  /// Additive-free mask: logits [V] -> logits with disallowed tokens at -inf.
  public func apply(_ logits: MLXArray) throws -> MLXArray {
    let t = Date(); defer { maskS += Date().timeIntervalSince(t) }
    if ended || after != nil { return logits }   // EOS already consumed, or grammar not started yet
    guard var w = try c.mask() else { return logits }
    let e = kit.eos
    if !w.contains(where: { $0 != 0 }) { throw ZError("grammar dead end: no token allowed") }
    if (w[e >> 5] >> Int32(e & 31)) & 1 == 1 { for s in kit.stop { w[s >> 5] |= Int32(truncatingIfNeeded: 1 << (s & 31)) } }
    let bits = ((MLXArray(w)[0..., .newAxis] >> shifts) & 1).flattened()[..<logits.dim(-1)]
    return which(bits .== 1, logits, MLXArray(-Float.infinity))
  }

  /// is `id` currently sampleable (stop tokens included)?
  public func allows(_ id: Int) throws -> Bool {
    if ended || after != nil { return true }
    guard let m = try c.mask() else { return true }
    func bit(_ i: Int) -> Bool { (m[i >> 5] >> Int32(i & 31)) & 1 == 1 }
    return bit(id) || (kit.stop.contains(id) && bit(kit.eos))
  }

  public func commit(_ id: Int) throws {
    let t = Date(); defer { maskS += Date().timeIntervalSince(t) }
    if ended { return }
    if let a = after { if id == a { after = nil }; return }
    if kit.stop.contains(id) { ended = true; return }   // matcher is "stopped" before EOS is sampled; only now is the output over
    committed += try c.commit(id)
  }
}

extension GrammarConstraint: Constraint {
  func mask() throws -> [Int32]? { let m = try computeMask(); return m.needsApply ? m.mask : nil }
  func commit(_ id: Int) throws -> Int { 1 + (try commitToken(Int32(id))).tokens.count }
  func rollback(_ n: Int) throws { try rollback(Int32(n)) }
}

extension GrammarKit {
  /// `schema`: `tag` is a JSON-schema source (response_format) instead of a structural tag.
  public func gate(tag: String, schema: Bool = false, lang: String? = nil, after: Int? = nil) throws -> (Gate, compile: Double) {
    let (c, k, t) = try take(tag, schema: schema, lang: lang)
    let g = Gate(c, kit: self, key: k); g.after = after
    return (g, t)
  }
  public func done(_ g: Gate, tag: String) { release(g.c, key: g.key, committed: g.committed) }
}
