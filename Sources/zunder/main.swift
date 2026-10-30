import Foundation
import ZunderCore

let usage = """
usage: zunder <command> [options]
  serve  --model <dir|hf-id> [--port N] [--idle 600] [--no-lease] [--ram GB] [--preload] [--dtype fp16|bf16|auto]   run in foreground
  start  (same options)    run in background, log ~/.local/state/zunder/zunder.log
  stop | status | log [-f]
  bench  --model <dir|hf-id> [--tokens 128] [--prompt-tokens 512] [--no-lease] [--ram GB] [--dtype ...]
  (env ZUNDER_DTYPE; auto = fp16 on M1/M2 GPUs)
model: local mlx-swift-lm text model dir, or org/name from the HF cache (never downloads).
"""

func die(_ s: String) -> Never { FileHandle.standardError.write(Data((s + "\n").utf8)); exit(1) }
var args = Array(CommandLine.arguments.dropFirst())
guard let cmd = args.first else { print(usage); exit(0) }
args.removeFirst()
func opt(_ k: String) -> String? { args.firstIndex(of: k).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
func flag(_ k: String) -> Bool { args.contains(k) }

let state = URL(fileURLWithPath: NSHomeDirectory() + "/.local/state/zunder")
try? FileManager.default.createDirectory(at: state, withIntermediateDirectories: true)
let pidFile = state.appendingPathComponent("pid"), logFile = state.appendingPathComponent("zunder.log"), portFile = state.appendingPathComponent("port")
func run(_ exe: String, _ a: [String]) -> String {
  let p = Process(), o = Pipe()
  p.executableURL = URL(fileURLWithPath: exe); p.arguments = a; p.standardOutput = o; p.standardError = FileHandle.nullDevice
  guard (try? p.run()) != nil else { return "" }
  let d = o.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit()
  return String(decoding: d, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
}
func portwart(_ a: [String]) -> String { run(NSHomeDirectory() + "/.local/bin/portwart", a) }
func claimPort() -> UInt16 {
  if let p = UInt16(portwart(["port", "zunder"])) { return p }
  if let d = try? JSONSerialization.jsonObject(with: Data(portwart(["claim", "zunder", "--health", "/health", "--proc", "zunder"]).utf8)) as? [String: Any], let p = d["result"] as? Int { return UInt16(p) }
  die("portwart claim zunder failed (use --port)")
}
func livePid() -> Int32? {
  guard let s = try? String(contentsOf: pidFile, encoding: .utf8), let p = Int32(s.trimmingCharacters(in: .whitespacesAndNewlines)), kill(p, 0) == 0 else { return nil }
  return p
}
func health(_ port: UInt16) -> [String: Any]? {
  var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/health")!); req.timeoutInterval = 2
  nonisolated(unsafe) var out: [String: Any]?
  let sem = DispatchSemaphore(value: 0)
  URLSession.shared.dataTask(with: req) { d, _, _ in out = d.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }; sem.signal() }.resume()
  sem.wait(); return out
}

func makeEngine() async throws -> Engine {
  guard let m = opt("--model") ?? ProcessInfo.processInfo.environment["ZUNDER_MODEL"] else { die("--model required") }
  let dir = try resolveModel(m)
  var lease: GPULease?
  if !flag("--no-lease") { lease = GPULease(label: "zunder " + dir.lastPathComponent, ram: Int(opt("--ram") ?? "") ?? Engine.ramGB(dir)) }
  return try await Engine(spec: m, lease: lease, dtype: opt("--dtype"))
}

switch cmd {
case "serve":
  let port = UInt16(opt("--port") ?? "") ?? claimPort()
  let eng = try await makeEngine()
  try? "\(getpid())".write(to: pidFile, atomically: true, encoding: .utf8)
  try? "\(port)".write(to: portFile, atomically: true, encoding: .utf8)
  let srv = Server(engine: eng, port: port, idle: Double(opt("--idle") ?? "") ?? 600)
  var sigs: [DispatchSourceSignal] = []
  for s in [SIGTERM, SIGINT] {
    signal(s, SIG_IGN)
    // own queue: srv.q may be blocked waiting for a gpu-run slot or inside a generation
    let src = DispatchSource.makeSignalSource(signal: s, queue: .global())
    src.setEventHandler { eng.lease?.release(); try? FileManager.default.removeItem(at: pidFile); exit(0) }   // frees the gpu-run slot; process exit frees the weights
    src.resume(); sigs.append(src)
  }
  if flag("--preload") { srv.q.async { do { try eng.ensureLoaded() } catch { eng.log("preload failed: \(error)") } } }
  try await srv.run()
case "start":
  if let p = livePid() { die("already running (pid \(p))") }
  FileManager.default.createFile(atPath: logFile.path, contents: nil)
  let port = UInt16(opt("--port") ?? "") ?? claimPort()
  var a = ["serve"] + args
  if opt("--port") == nil { a += ["--port", String(port)] }
  let p = Process()
  p.executableURL = URL(fileURLWithPath: CommandLine.arguments[0].hasPrefix("/") ? CommandLine.arguments[0] : FileManager.default.currentDirectoryPath + "/" + CommandLine.arguments[0])
  p.arguments = a
  let lh = try FileHandle(forWritingTo: logFile); lh.seekToEndOfFile()
  p.standardOutput = lh; p.standardError = lh; p.standardInput = FileHandle.nullDevice
  try p.run()
  for _ in 0 ..< 60 { if health(port) != nil { print("zunder up on http://127.0.0.1:\(port)/v1 (pid \(p.processIdentifier))"); exit(0) }; Thread.sleep(forTimeInterval: 0.5) }
  die("did not come up, see \(logFile.path)")
case "stop":
  guard let p = livePid() else { print("not running"); exit(0) }
  kill(p, SIGTERM)
  for _ in 0 ..< 60 { if kill(p, 0) != 0 { print("stopped"); try? FileManager.default.removeItem(at: pidFile); exit(0) }; Thread.sleep(forTimeInterval: 0.5) }
  die("pid \(p) still alive after 30 s (not killed)")
case "status":
  guard let p = livePid() else { print("not running"); exit(1) }
  let port = (try? String(contentsOf: portFile, encoding: .utf8)).flatMap { UInt16($0) } ?? 0
  print("pid \(p) port \(port) " + (health(port).map { String(decoding: jdump($0), as: UTF8.self) } ?? "no /health answer"))
case "log":
  let t = Process(); t.executableURL = URL(fileURLWithPath: "/usr/bin/tail"); t.arguments = (flag("-f") ? ["-f"] : []) + ["-n", "50", logFile.path]
  try t.run(); t.waitUntilExit()
case "bench":
  let eng = try await makeEngine()
  let nPrompt = Int(opt("--prompt-tokens") ?? "") ?? 512, nGen = Int(opt("--tokens") ?? "") ?? 128
  try eng.ensureLoaded()
  defer { eng.unload() }
  let text = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: nPrompt / 8 + 1)
  let ids = Array(eng.tok.encode(text: text, addSpecialTokens: false).prefix(nPrompt))
  let tool: JSON = ["type": "function", "function": ["name": "set_mode", "parameters": ["type": "object", "properties": ["mode": ["type": "string", "enum": ["a", "b"]]], "required": ["mode"]] as JSON] as JSON]
  _ = try eng.generate(ids: Array(ids.prefix(64)), tag: nil, params: SampleParams(temp: 0), maxTokens: 8) { _ in true }   // warm kernels
  let free = try eng.generate(ids: ids, tag: nil, params: SampleParams(temp: 0), maxTokens: nGen) { _ in true }
  print("unconstrained: " + free.stats.line)
  let tag = try eng.fmt.structuralTag(tools: [tool], GrammarOpts(required: true, think: false, parallel: false))
  _ = try eng.generate(ids: ids, tag: tag, params: SampleParams(), maxTokens: 4) { _ in true }   // warm the compile cache
  let g = try eng.generate(ids: ids, tag: tag, params: SampleParams(), maxTokens: nGen) { _ in true }
  print("grammar:       " + g.stats.line)
  eng.checkFinite = true
  _ = try eng.generate(ids: ids, tag: nil, params: SampleParams(temp: 0), maxTokens: min(nGen, 48)) { _ in true }
  print("dtype check: non-finite logit steps = \(eng.nonFinite)")
case "gen":   // greedy parity dump: --prompts file.json (list of strings) --tokens N --out file.json
  let eng = try await makeEngine()
  let n = Int(opt("--tokens") ?? "") ?? 32
  guard let pf = opt("--prompts"), let po = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: pf)))  as? [String] else { die("--prompts file.json (list of strings)") }
  try eng.ensureLoaded()
  defer { eng.unload() }
  eng.checkFinite = true
  var res: [[String: Any]] = []
  for p in po {
    let ids = try eng.prompt(messages: [["role": "user", "content": p]], tools: nil)
    var tr: [[String: Any]] = []
    eng.trace = { tr.append(["ids": $0, "logits": $1.map { $0.isFinite ? $0 : -1e9 }]) }
    let o = try eng.generate(ids: ids, tag: nil, params: SampleParams(temp: 0), maxTokens: n) { _ in true }
    eng.trace = nil
    res.append(["prompt": p, "prompt_ids": ids, "tokens": o.tokens, "top8": tr, "stats": o.stats.line, "nonfinite": eng.nonFinite])
    print(o.stats.line)
  }
  try JSONSerialization.data(withJSONObject: res).write(to: URL(fileURLWithPath: opt("--out") ?? "gen.json"))
default: print(usage)
}
