import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon

/// xgrammar vocab for one tokenizer + compiled-constraint cache (matchers are reused via rollback).
public final class GrammarKit {
  public let gtok: GrammarTokenizer
  public let eos: Int
  public let stop: Set<Int>
  var cache: [String: GrammarConstraint] = [:]

  public init(tok: any MLXLMCommon.Tokenizer, eos: Int, stop: Set<Int>) throws {
    let v = TokenizerVocabExtractor.extractForGrammar(from: tok)
    gtok = try GrammarTokenizer(vocab: v.vocab, vocabType: v.vocabType, eosTokenId: Int32(eos))
    self.eos = eos; self.stop = stop.union([eos])
  }

  public var vocab: Int { gtok.vocabSize }

  /// Fresh-state matcher for `tag`; give it back with `release` so the compile is reused.
  public func take(_ tag: String) throws -> (GrammarConstraint, compile: Double) {
    if let c = cache.removeValue(forKey: tag) { return (c, 0) }
    let t = Date()
    let c = try GrammarConstraint(tokenizer: gtok, structuralTag: tag)
    return (c, Date().timeIntervalSince(t))
  }
  public func release(_ c: GrammarConstraint, tag: String, committed: Int) {
    if committed > 0 { do { try c.rollback(Int32(committed)) } catch { return } }
    cache[tag] = c
  }
  public func clear() { cache = [:] }
}

/// Per-request mask state: wraps a matcher, lets every stop token through when the grammar allows EOS.
public final class Gate {
  let c: GrammarConstraint, kit: GrammarKit
  public private(set) var committed = 0
  public var maskS = 0.0
  var shifts = MLXArray(Array(0 ..< 32).map { Int32($0) })
  init(_ c: GrammarConstraint, kit: GrammarKit) { self.c = c; self.kit = kit }

  /// Additive-free mask: logits [V] -> logits with disallowed tokens at -inf.
  public func apply(_ logits: MLXArray) throws -> MLXArray {
    let t = Date(); defer { maskS += Date().timeIntervalSince(t) }
    let m = try c.computeMask()
    if !m.needsApply { return logits }
    var w = m.mask
    let e = kit.eos
    if (w[e >> 5] >> Int32(e & 31)) & 1 == 1 { for s in kit.stop { w[s >> 5] |= Int32(truncatingIfNeeded: 1 << (s & 31)) } }
    let bits = ((MLXArray(w)[0..., .newAxis] >> shifts) & 1).flattened()[..<logits.dim(-1)]
    return which(bits .== 1, logits, MLXArray(-Float.infinity))
  }

  /// is `id` currently sampleable (stop tokens included)?
  public func allows(_ id: Int) throws -> Bool {
    let m = try c.computeMask()
    if !m.needsApply { return true }
    func bit(_ i: Int) -> Bool { (m.mask[i >> 5] >> Int32(i & 31)) & 1 == 1 }
    return bit(id) || (kit.stop.contains(id) && bit(kit.eos))
  }

  public func commit(_ id: Int) throws {
    let t = Date(); defer { maskS += Date().timeIntervalSince(t) }
    let r = try c.commitToken(Int32(id)); committed += 1 + r.tokens.count
  }
}

extension GrammarKit {
  public func gate(tag: String) throws -> (Gate, compile: Double) {
    let (c, t) = try take(tag)
    return (Gate(c, kit: self), t)
  }
  public func done(_ g: Gate, tag: String) { release(g.c, tag: tag, committed: g.committed) }
}
