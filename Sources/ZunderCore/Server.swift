import Foundation
import Network

public func jdump(_ o: Any) -> Data { (try? JSONSerialization.data(withJSONObject: o, options: [.withoutEscapingSlashes])) ?? Data("{}".utf8) }

/// OpenAI-compatible server, 127.0.0.1 only, one generation at a time.
public final class Server {
  let eng: Engine, port: UInt16, idle: TimeInterval
  public let q = DispatchQueue(label: "zunder.gen")
  let lock = NSLock()
  var busy = 0
  public var defaults = SampleParams()

  public init(engine: Engine, port: UInt16, idle: TimeInterval) { eng = engine; self.port = port; self.idle = idle }

  func hdr(_ head: String, _ k: String) -> String? {
    for l in head.components(separatedBy: "\r\n").dropFirst() {
      guard let i = l.firstIndex(of: ":") else { continue }
      if l[..<i].lowercased() == k { return l[l.index(after: i)...].trimmingCharacters(in: .whitespaces) }
    }
    return nil
  }

  public func run() async throws {
    let params = NWParameters.tcp
    params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: port)!)
    let l = try NWListener(using: params)
    l.newConnectionHandler = { [self] c in
      c.start(queue: .global())
      var buf = Data()
      func recv() {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { d, _, done, e in
          if let d { buf.append(d) }
          if let r = buf.range(of: Data("\r\n\r\n".utf8)) {
            let head = String(decoding: buf[..<r.lowerBound], as: UTF8.self)
            let len = max(0, Int(self.hdr(head, "content-length") ?? "") ?? 0)
            if len > 64 << 20 { return self.reply(c, 413, ["error": ["message": "payload too large"]]) }
            if buf.count - r.upperBound < len { if !done && e == nil { recv() } else { c.cancel() }; return }
            self.handle(c, head, Data(buf[r.upperBound ..< (r.upperBound + len)]))
          } else if !done && e == nil { recv() } else { c.cancel() }
        }
      }
      recv()
    }
    l.start(queue: .global())
    eng.log("zunder: http://127.0.0.1:\(port)/v1  model \(eng.id)  idle \(Int(idle)) s")
    while true {
      try await Task.sleep(nanoseconds: 5_000_000_000)
      let b = isBusy()
      if !b, eng.loaded, Date().timeIntervalSince(eng.lastUse) > idle { q.async { if Date().timeIntervalSince(self.eng.lastUse) > self.idle { self.eng.unload() } } }
    }
  }

  func isBusy() -> Bool { lock.lock(); defer { lock.unlock() }; return busy > 0 }

  func reply(_ c: NWConnection, _ code: Int, _ body: Any) {
    let b = jdump(body)
    let h = "HTTP/1.1 \(code) \(code == 200 ? "OK" : "Error")\r\nContent-Type: application/json\r\nContent-Length: \(b.count)\r\nConnection: close\r\n\r\n"
    c.send(content: Data(h.utf8) + b, completion: .contentProcessed { _ in c.cancel() })
  }

  func handle(_ c: NWConnection, _ head: String, _ body: Data) {
    let parts = head.split(separator: "\r\n").first?.split(separator: " ") ?? []
    guard parts.count >= 2 else { return c.cancel() }
    let method = String(parts[0]), path = String(parts[1].split(separator: "?").first ?? "")
    // DNS-rebinding / CSRF guard: loopback Host only, no foreign browser Origin
    let host = (hdr(head, "host") ?? "").replacingOccurrences(of: ":\(port)", with: "")
    guard ["127.0.0.1", "localhost", "[::1]"].contains(host) else { return reply(c, 403, ["error": ["message": "host"]]) }
    if let o = hdr(head, "origin"), !["127.0.0.1", "localhost", "[::1]"].contains(URL(string: o)?.host ?? "") { return reply(c, 403, ["error": ["message": "origin"]]) }
    switch (method, path) {
    case ("GET", "/health"):
      lock.lock(); let b = busy; lock.unlock()
      reply(c, 200, ["status": "ok", "model": eng.id, "loaded": eng.loaded, "busy": b > 0, "idle_s": Int(Date().timeIntervalSince(eng.lastUse))])
    case ("GET", "/v1/models"):
      reply(c, 200, ["object": "list", "data": [["id": eng.id, "object": "model", "created": 0, "owned_by": "zunder"]]])
    case ("POST", "/v1/chat/completions"):
      guard let req = (try? JSONSerialization.jsonObject(with: body)) as? JSON else { return reply(c, 400, ["error": ["message": "invalid JSON"]]) }
      lock.lock(); busy += 1; lock.unlock()
      q.async { [self] in
        defer { lock.lock(); busy -= 1; lock.unlock() }
        chat(c, req)
      }
    default: reply(c, 404, ["error": ["message": "not found"]])
    }
  }

  func norm(_ ms: [JSON]) -> [JSON] {
    ms.map { m in
      var o = m
      if let parts = m["content"] as? [JSON] { o["content"] = parts.compactMap { $0["text"] as? String }.joined() }
      if o["content"] is NSNull { o["content"] = nil }
      if o["reasoning_content"] == nil, let r = m["reasoning"] as? String { o["reasoning_content"] = r }
      if let tcs = m["tool_calls"] as? [JSON] {
        o["tool_calls"] = tcs.map { tc in
          var t = tc
          if var f = tc["function"] as? JSON, let a = f["arguments"] as? String {
            f["arguments"] = (try? JSONSerialization.jsonObject(with: Data(a.utf8))) as? JSON ?? [:]
            t["function"] = f
          }
          return t
        }
      }
      return o
    }
  }

  func chat(_ c: NWConnection, _ req: JSON) {
    let stream = req["stream"] as? Bool ?? false
    let id = "chatcmpl-" + UUID().uuidString.prefix(12).lowercased()
    let model = req["model"] as? String ?? eng.id
    func fail(_ code: Int, _ msg: String) {
      eng.log("error: \(msg)")
      if stream { c.send(content: Data("data: \(String(decoding: jdump(["error": ["message": msg]]), as: UTF8.self))\n\n".utf8), completion: .contentProcessed { _ in c.cancel() }) }
      else { reply(c, code, ["error": ["message": msg]]) }
    }
    do {
      var tools = (req["tools"] as? [JSON]) ?? []
      var required = false
      switch req["tool_choice"] {
      case let s as String where s == "none": tools = []
      case let s as String where s == "required": required = true
      case let o as JSON:
        if let n = (o["function"] as? JSON)?["name"] as? String { tools = tools.filter { (($0["function"] as? JSON) ?? $0)["name"] as? String == n }; required = true }
      default: break
      }
      var p = defaults
      if let v = (req["temperature"] as? NSNumber)?.floatValue { p.temp = v }
      if let v = (req["top_p"] as? NSNumber)?.floatValue { p.topP = v }
      if let v = (req["top_k"] as? NSNumber)?.intValue { p.topK = v }
      let maxTok = (req["max_completion_tokens"] as? Int) ?? (req["max_tokens"] as? Int) ?? 4096
      let think = (req["think"] as? Bool) ?? ((req["chat_template_kwargs"] as? JSON)?["enable_thinking"] as? Bool) ?? true
      let useGrammar = !tools.isEmpty && (req["grammar"] as? String) != "none"

      try eng.ensureLoaded()
      let ids = try eng.prompt(messages: norm(req["messages"] as? [JSON] ?? []), tools: tools.isEmpty ? nil : tools)
      let tag = useGrammar ? try eng.fmt.structuralTag(tools: tools, GrammarOpts(required: required, think: think, parallel: (req["parallel_tool_calls"] as? Bool) ?? true)) : nil
      let parser = eng.fmt.parser(tools: tools.isEmpty ? nil : tools)
      var detok = Detok(eng.tok)
      var calls = 0, reasoning = "", content = "", toolCalls: [JSON] = []
      let created = Int(Date().timeIntervalSince1970)

      func chunk(_ delta: JSON, finish: String? = nil, usage: JSON? = nil) -> Data {
        var o: JSON = ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
          "choices": usage != nil ? [] : [["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()] as JSON]]
        if let usage { o["usage"] = usage }
        return Data("data: ".utf8) + jdump(o) + Data("\n\n".utf8)
      }
      func emit(_ evs: [Event]) {
        for e in evs {
          switch e {
          case .reasoning(let s): reasoning += s; if stream { c.send(content: chunk(["reasoning_content": s]), completion: .idempotent) }
          case .content(let s): content += s; if stream { c.send(content: chunk(["content": s]), completion: .idempotent) }
          case .call(let name, let args):
            let tc: JSON = ["id": "call_" + UUID().uuidString.prefix(8).lowercased(), "type": "function", "function": ["name": name, "arguments": args] as JSON]
            toolCalls.append(tc)
            if stream { var d = tc; d["index"] = calls; c.send(content: chunk(["tool_calls": [d]]), completion: .idempotent) }
            calls += 1
          }
        }
      }
      if stream {
        let h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        c.send(content: Data(h.utf8) + chunk(["role": "assistant", "content": ""]), completion: .idempotent)
      }
      nonisolated(unsafe) var gone = false
      c.stateUpdateHandler = { if case .cancelled = $0 { gone = true } else if case .failed = $0 { gone = true } }
      let out = try eng.generate(ids: ids, tag: tag, params: p, maxTokens: maxTok) { t in
        emit(parser.feed(detok.push(t)))
        return !gone
      }
      emit(parser.finish())
      eng.log("\(id) \(out.finish) · \(out.stats.line)\(tag != nil ? " · grammar on" : "")")
      let finish = calls > 0 ? "tool_calls" : out.finish == "length" ? "length" : "stop"
      let usage: JSON = ["prompt_tokens": ids.count, "completion_tokens": out.tokens.count, "total_tokens": ids.count + out.tokens.count]
      if stream {
        var tail = chunk([:], finish: finish)
        if (req["stream_options"] as? JSON)?["include_usage"] as? Bool == true { tail += chunk([:], usage: usage) }
        tail += Data("data: [DONE]\n\n".utf8)
        c.send(content: tail, contentContext: .defaultMessage, isComplete: true, completion: .contentProcessed { _ in c.cancel() })
      } else {
        var msg: JSON = ["role": "assistant", "content": content.isEmpty && !toolCalls.isEmpty ? NSNull() : content]
        if !reasoning.isEmpty { msg["reasoning_content"] = reasoning }
        if !toolCalls.isEmpty { msg["tool_calls"] = toolCalls }
        reply(c, 200, ["id": id, "object": "chat.completion", "created": created, "model": model,
          "choices": [["index": 0, "message": msg, "finish_reason": finish] as JSON], "usage": usage,
          "timings": ["prefill_tps": out.stats.prefillTps, "decode_tps": out.stats.decTps, "mask_us_per_token": out.stats.maskUs]])
      }
    } catch { fail(500, "\(error)") }
  }
}
