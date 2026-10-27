// Grammar construction ported from recursive-oss/rec (engine/src/rec_mlx/grammar_builder.py, Apache-2.0); see NOTICE.
import Foundation

public typealias JSON = [String: Any]

public enum Event: Equatable {
  case reasoning(String), content(String), call(name: String, args: String)
}

public protocol ToolFormat: Sendable {
  /// xgrammar structural-tag JSON: optional reasoning, free text, tool calls from `tools` (OpenAI tool objects).
  func structuralTag(tools: [JSON]) throws -> String
  func parser(tools: [JSON]?) -> ToolStreamParser
}

public protocol ToolStreamParser: AnyObject {
  func feed(_ s: String) -> [Event]
  func finish() -> [Event]
}

public func toolFormat(modelType: String) -> any ToolFormat { GLM47() }

public struct GLM47: ToolFormat {
  static let markers = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
    "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>", "<|user|>", "<|assistant|>", "<|observation|>"]
  static let triggers = ["<think>", "<tool_call>"]
  public init() {}

  public func structuralTag(tools: [JSON]) throws -> String {
    let trig = tools.isEmpty ? ["<think>"] : Self.triggers
    let free = Self.markers.filter { !trig.contains($0) }
    let think: JSON = ["type": "tag", "begin": "<think>", "end": "</think>",
      "content": ["type": "any_text", "excludes": Self.markers.filter { $0 != "</think>" }] as JSON]
    var tags: [JSON] = [think]
    for t in tools {
      guard let f = t["function"] as? JSON ?? (t["name"] != nil ? t : nil), let name = f["name"] as? String else { throw ZError("tool without name") }
      let ps = f["parameters"] as? JSON ?? ["type": "object", "properties": JSON()]
      guard ps["type"] as? String ?? "object" == "object" else { throw ZError("tool \(name): parameters must be an object schema") }
      tags.append(["type": "tag", "begin": "<tool_call>" + name, "end": "</tool_call>", "content": body(ps)])
    }
    let fmt: JSON = ["type": "triggered_tags", "triggers": trig, "tags": tags, "excludes": free]
    let data = try JSONSerialization.data(withJSONObject: ["type": "structural_tag", "format": fmt] as JSON, options: [.withoutEscapingSlashes])
    return String(decoding: data, as: UTF8.self)
  }

  /// `<arg_key>k</arg_key><arg_value>v</arg_value>` per property: required ones in `required` order, optional ones as an unordered loop.
  func body(_ ps: JSON) -> JSON {
    let props = ps["properties"] as? JSON ?? [:]
    let req = (ps["required"] as? [String] ?? []).filter { props[$0] != nil }
    let opt = props.keys.filter { !req.contains($0) }.sorted()
    func arg(_ k: String) -> JSON {
      let sch = props[k] as? JSON ?? [:]
      let content: JSON
      if let e = sch["enum"] as? [String], !e.isEmpty { content = ["type": "or", "elements": e.map { ["type": "const_string", "value": $0] as JSON }] }   // raw (unquoted) string values
      else if sch["type"] as? String == "string" { content = ["type": "any_text", "excludes": Self.markers.filter { $0 != "</arg_value>" }] }
      else { content = ["type": "json_schema", "json_schema": sch] }
      return ["type": "tag", "begin": "<arg_key>\(k)</arg_key><arg_value>", "end": "</arg_value>", "content": content]
    }
    var els = req.map(arg)
    if !opt.isEmpty { els.append(["type": "tags_with_separator", "tags": opt.map(arg), "separator": ""]) }
    if els.isEmpty { return ["type": "any_text", "excludes": Self.markers] }
    return els.count == 1 ? els[0] : ["type": "sequence", "elements": els]
  }

  public func parser(tools: [JSON]?) -> ToolStreamParser { GLMParser(tools: tools) }
}

/// Incremental splitter for `<think>..</think>`, free text and `<tool_call>..</tool_call>`. Does not assume reasoning comes first.
final class GLMParser: ToolStreamParser {
  enum St { case content, reasoning, call }
  var st = St.content, buf = ""
  let tools: [JSON]?
  init(tools: [JSON]?) { self.tools = tools }

  func feed(_ s: String) -> [Event] { buf += s; return drain(final: false) }
  func finish() -> [Event] { drain(final: true) }

  func drain(final: Bool) -> [Event] {
    var out: [Event] = []
    while true {
      let marks: [(String, St)] = st == .content ? [("<think>", .reasoning), ("<tool_call>", .call)] : st == .reasoning ? [("</think>", .content)] : [("</tool_call>", .content)]
      let hit = marks.compactMap { m in buf.range(of: m.0).map { ($0, m) } }.min { $0.0.lowerBound < $1.0.lowerBound }
      if let (r, m) = hit {
        let head = String(buf[..<r.lowerBound])
        buf = String(buf[r.upperBound...])
        if st == .call { out += callEvent(head) } else if !head.isEmpty { out.append(st == .reasoning ? .reasoning(head) : .content(head)) }
        st = m.1
        continue
      }
      if st == .call { if final { out += callEvent(buf, truncated: true); buf = "" }; return out }
      // hold back a tail that may be the start of a marker
      var keep = 0
      if !final { for m in marks { for n in stride(from: min(m.0.count - 1, buf.count), to: 0, by: -1) where buf.hasSuffix(String(m.0.prefix(n))) { keep = max(keep, n); break } } }
      let emit = String(buf.dropLast(keep))
      buf = String(buf.suffix(keep))
      if !emit.isEmpty { out.append(st == .reasoning ? .reasoning(emit) : .content(emit)) }
      return out
    }
  }

  func callEvent(_ s: String, truncated: Bool = false) -> [Event] {
    let raw = "<tool_call>" + s + (truncated ? "" : "</tool_call>")
    guard !truncated, let k = s.range(of: "<arg_key>") ?? (s.contains("<") ? nil : s.endIndex ..< s.endIndex) else { return [.content(raw)] }
    let name = s[..<k.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty else { return [.content(raw)] }
    var parts: [String] = [], rest = s[k.lowerBound...]
    while let a = rest.range(of: "<arg_key>"), let b = rest.range(of: "</arg_key>"), let c = rest.range(of: "<arg_value>"), let d = rest.range(of: "</arg_value>") {
      let key = rest[a.upperBound ..< b.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
      let val = rest[c.upperBound ..< d.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
      let isStr = schemaType(name, key) == "string"
      let lit = !isStr && (try? JSONSerialization.jsonObject(with: Data(val.utf8), options: .fragmentsAllowed)) != nil ? val : jsonStr(val)
      parts.append(jsonStr(key) + ":" + lit)
      rest = rest[d.upperBound...]
    }
    return [.call(name: name, args: "{" + parts.joined(separator: ",") + "}")]
  }

  func schemaType(_ fn: String, _ arg: String) -> String? {
    for t in tools ?? [] {
      let f = t["function"] as? JSON ?? t
      if f["name"] as? String == fn, let p = (f["parameters"] as? JSON)?["properties"] as? JSON { return (p[arg] as? JSON)?["type"] as? String }
    }
    return nil
  }
}

func jsonStr(_ s: String) -> String {
  String(decoding: (try? JSONSerialization.data(withJSONObject: s, options: [.fragmentsAllowed, .withoutEscapingSlashes])) ?? Data("\"\"".utf8), as: UTF8.self)
}
