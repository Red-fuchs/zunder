import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public struct GenStats {
  public var promptTok = 0, genTok = 0
  public var prefillS = 0.0, decS = 0.0, maskS = 0.0, compileS = 0.0
  public var prefillTps: Double { prefillS > 0 ? Double(promptTok) / prefillS : 0 }
  public var decTps: Double { decS > 0 ? Double(genTok) / decS : 0 }
  public var maskUs: Double { genTok > 0 ? maskS * 1e6 / Double(genTok) : 0 }
  public var line: String {
    String(format: "prefill %d tok %.2fs (%.0f tok/s) · gen %d tok %.2fs (%.1f tok/s) · mask %.0f µs/tok · compile %.2fs", promptTok, prefillS, prefillTps, genTok, decS, decTps, maskUs, compileS)
  }
}

/// One loaded model + tokenizer + grammar vocab. Calls are serialised by the caller (single request at a time).
public final class Engine {
  public let id: String, dir: URL
  public let tok: any MLXLMCommon.Tokenizer
  public let fmt: any ToolFormat
  public let modelType: String
  public var lease: GPULease?
  /// Parity tracing: called with top-8 (ids, raw logits) of every step.
  public var trace: (([Int], [Float]) -> Void)?
  public var log: (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
  public private(set) var lastUse = Date()
  var ctx: ModelContext?
  var kit: GrammarKit?
  let stop: Set<Int>, eos: Int
  let vocabSize: Int?

  public var loaded: Bool { ctx != nil }

  public init(spec: String, lease: GPULease? = nil) async throws {
    let dir = try resolveModel(spec)
    self.dir = dir
    id = FileManager.default.fileExists(atPath: (spec as NSString).expandingTildeInPath) ? dir.lastPathComponent : spec
    let tok = try await loadTokenizer(dir)
    self.tok = tok
    self.lease = lease
    let cfg = (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("config.json")))) as? JSON ?? [:]
    modelType = cfg["model_type"] as? String ?? ""
    vocabSize = cfg["vocab_size"] as? Int
    fmt = toolFormat(modelType: modelType)
    var s = Set<Int>()
    for f in ["config.json", "generation_config.json"] {
      let j = (try? JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(f)))) as? JSON
      if let e = j?["eos_token_id"] as? Int { s.insert(e) } else if let e = j?["eos_token_id"] as? [Int] { s.formUnion(e) }
    }
    eos = tok.eosToken.flatMap { tok.convertTokenToId($0) } ?? s.min() ?? 0
    stop = s.union([eos])
  }

  /// Approximate resident size for the gpu-run RAM gate.
  public static func ramGB(_ dir: URL) -> Int {
    let n = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? [])
      .filter { $0.pathExtension == "safetensors" }.reduce(0) { $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    return Int(ceil(Double(n) * 1.1 / 1e9)) + 2
  }

  public func ensureLoaded() throws {
    lastUse = Date()
    if ctx != nil { return }
    try lease?.acquire(log: log)
    let t = Date(), d = dir
    nonisolated(unsafe) var res: Result<ModelContext, Error> = .failure(ZError("load"))
    let sem = DispatchSemaphore(value: 0)
    Task { do { res = .success(try await loadContext(d)) } catch { res = .failure(error) }; sem.signal() }
    sem.wait()
    do { let c = try res.get(); kit = try GrammarKit(tok: tok, eos: eos, stop: stop, dir: dir, logitDim: vocabSize); ctx = c } catch { kit = nil; lease?.release(); throw error }
    log(String(format: "model loaded %@ (%.1f s)", id, Date().timeIntervalSince(t)))
    lastUse = Date()
  }

  public func unload() {
    guard ctx != nil else { return }
    ctx = nil; kit = nil
    MLX.GPU.clearCache()
    lease?.release()
    log("model unloaded")
  }

  public func prompt(messages: [JSON], tools: [JSON]?) throws -> [Int] {
    try tok.applyChatTemplate(messages: messages.map(sendable), tools: tools?.map(sendable), additionalContext: nil)
  }

  public struct Out { public var tokens: [Int]; public var finish: String; public var stats: GenStats }

  /// `tag` = structural-tag grammar, or JSON-schema source if `schema` (nil = unconstrained). `onToken` returns false to cancel.
  public func generate(ids: [Int], tag: String?, schema: Bool = false, params: SampleParams, maxTokens: Int, onToken: (Int) -> Bool) throws -> Out {
    try ensureLoaded()
    guard let ctx, let kit else { throw ZError("not loaded") }
    var st = GenStats()
    st.promptTok = ids.count
    var gate: Gate?
    if let tag { let (g, c) = try kit.gate(tag: tag, schema: schema); gate = g; st.compileS = c }
    defer { if let g = gate, let tag { kit.done(g, tag: tag) } }

    let cache = try ctx.model.newCache(parameters: nil)
    let t0 = Date()
    var logits: MLXArray
    let inp = LMInput(tokens: MLXArray(ids.map { Int32($0) }))
    switch try ctx.model.prepare(inp, cache: cache, state: nil, prefill: .init(stepSize: 512)) {
    case .tokens(let t): logits = ctx.model(t[text: .newAxis], cache: cache, state: nil).logits
    case .logits(let o): logits = o.logits
    }
    eval(logits)
    st.prefillS = Date().timeIntervalSince(t0)

    var out: [Int] = [], finish = "length"
    let t1 = Date()
    func step(_ l: MLXArray) throws -> MLXArray {
      var l = l[0, -1, 0...]
      if let trace { let i = argSort(-l.asType(.float32), axis: -1)[..<8]; let v = takeAlong(l.asType(.float32), i, axis: -1); trace(i.asArray(Int.self), v.asArray(Float.self)) }
      if let gate { l = try gate.apply(l) }
      return sample(l, params)
    }
    func fwd(_ y: MLXArray) -> MLXArray { ctx.model(y.reshaped(1, 1), cache: cache) }
    var y = try step(logits)
    asyncEval(y)
    while out.count < maxTokens {
      var next: MLXArray?
      if gate == nil, out.count + 1 < maxTokens { let nl = fwd(y); next = try step(nl); asyncEval(next!) }   // unconstrained: overlap next step with the host work
      let t = y.item(Int.self)
      if stop.contains(t) { finish = "stop"; break }
      try gate?.commit(t)
      out.append(t)
      if !onToken(t) { finish = "cancel"; break }
      if out.count >= maxTokens { break }
      if let next { y = next } else { y = try step(fwd(MLXArray([Int32(t)]))); asyncEval(y) }
    }
    st.genTok = out.count; st.decS = Date().timeIntervalSince(t1); st.maskS = gate?.maskS ?? 0
    lastUse = Date()
    return Out(tokens: out, finish: finish, stats: st)
  }
}

func sendable(_ x: JSON) -> [String: any Sendable] { x.mapValues(sendableAny) }
func sendableAny(_ x: Any) -> any Sendable {
  switch x {
  case let d as JSON: return sendable(d)
  case let a as [Any]: return a.map(sendableAny)
  case let s as String: return s
  case let n as NSNumber: return n
  default: return "\(x)"
  }
}

/// Token ids -> text (special tokens included, so the stream parser sees the markers).
public struct Detok {
  var d: NaiveStreamingDetokenizer
  public init(_ tok: any MLXLMCommon.Tokenizer) { d = NaiveStreamingDetokenizer(tokenizer: tok) }
  public mutating func push(_ t: Int) -> String { d.append(token: t); return d.next() ?? "" }
}
