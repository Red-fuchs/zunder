import Foundation

/// Holds a gpu-run slot only while a model is loaded. gpu-run has no acquire/release mode, so a tiny holder job runs under it
/// (FIFO queue + RAM gate); it exits when the keep file disappears or the owning process dies. Idea ported from ~/Projects/sl Lease.swift.
public final class GPULease {
  let label: String, ram: Int, timeout: TimeInterval
  var proc: Process?, dir: URL?
  public init(label: String, ram: Int, timeout: TimeInterval = 1800) { self.label = label; self.ram = ram; self.timeout = timeout }
  public var held: Bool { proc?.isRunning == true }

  static func gpuRun() -> String? {
    [NSHomeDirectory() + "/.local/bin/gpu-run", "/opt/homebrew/bin/gpu-run", "/usr/local/bin/gpu-run"].first { FileManager.default.isExecutableFile(atPath: $0) }
  }

  public func acquire(log: @escaping (String) -> Void) throws {
    if held { return }
    guard let g = Self.gpuRun() else { throw ZError("gpu-run not found") }
    let d = FileManager.default.temporaryDirectory.appendingPathComponent("zunder-lease-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    let keep = d.appendingPathComponent("keep").path, ready = d.appendingPathComponent("ready").path
    FileManager.default.createFile(atPath: keep, contents: Data())
    let p = Process()
    p.executableURL = URL(fileURLWithPath: g)
    let holder = #"echo $$ > "$1"; while [ -e "$2" ] && kill -0 "$3" 2>/dev/null; do sleep 1; done"#
    p.arguments = ["-n", label, "--ram", String(ram), "sh", "-c", holder, "zunder-holder", ready, keep, String(getpid())]
    let err = Pipe(); p.standardError = err; p.standardOutput = FileHandle.nullDevice
    err.fileHandleForReading.readabilityHandler = { h in
      let s = String(decoding: h.availableData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
      if !s.isEmpty { log("gpu-run: " + s) }
    }
    try p.run()
    proc = p; dir = d
    let t0 = Date()
    while !FileManager.default.fileExists(atPath: ready) {
      if !p.isRunning { err.fileHandleForReading.readabilityHandler = nil; cleanup(); throw ZError("gpu-run exited without a slot") }
      if Date().timeIntervalSince(t0) > timeout { err.fileHandleForReading.readabilityHandler = nil; release(); throw ZError("gpu-run slot timeout after \(Int(timeout)) s") }
      Thread.sleep(forTimeInterval: 0.3)
    }
    err.fileHandleForReading.readabilityHandler = nil
    log("gpu slot acquired (\(ram) GB)")
  }

  public func release() {
    guard let p = proc else { return }
    if let d = dir { try? FileManager.default.removeItem(at: d.appendingPathComponent("keep")) }
    let t0 = Date()
    while p.isRunning && Date().timeIntervalSince(t0) < 6 { Thread.sleep(forTimeInterval: 0.2) }
    if p.isRunning { p.terminate() }
    p.waitUntilExit()
    cleanup()
  }
  func cleanup() { if let d = dir { try? FileManager.default.removeItem(at: d) }; proc = nil; dir = nil }
  deinit { release() }
}
