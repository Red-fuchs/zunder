import XCTest
import MLX
import MLXLMCommon
@testable import ZunderCore

let tools: [JSON] = [
  ["type": "function", "function": ["name": "get_weather", "description": "w", "parameters": [
    "type": "object", "properties": ["city": ["type": "string"], "unit": ["type": "string", "enum": ["c", "f"]]], "required": ["city"]] as JSON] as JSON],
  ["type": "function", "function": ["name": "add", "parameters": [
    "type": "object", "properties": ["a": ["type": "integer"], "b": ["type": "integer"]], "required": ["a", "b"]] as JSON] as JSON],
]

final class GrammarTests: XCTestCase {
  static var tok: (any MLXLMCommon.Tokenizer)!
  static var kit: GrammarKit!
  static var tag = ""

  override class func setUp() {
    let sem = DispatchSemaphore(value: 0)
    Task {
      tok = try await loadTokenizer(root.appendingPathComponent("fixtures/flint-tok"))
      let stop = Set([150001, 150003, 150005, 150018])
      kit = try GrammarKit(tok: tok, eos: tok.convertTokenToId("<|end_of_text|>")!, stop: stop, dir: root.appendingPathComponent("fixtures/flint-tok"))
      tag = try GLM47().structuralTag(tools: tools)
      sem.signal()
    }
    sem.wait()
  }

  /// index of first token the grammar refuses (nil = all accepted)
  func firstReject(_ text: String, tag: String = GrammarTests.tag) throws -> Int? {
    let (g, _) = try Self.kit.gate(tag: tag)
    defer { Self.kit.done(g, tag: tag) }
    let ids = Self.tok.encode(text: text, addSpecialTokens: false)
    for (i, id) in ids.enumerated() {
      if try !g.allows(id) { return i }
      try g.commit(id)
    }
    return nil
  }

  /// response_format json_schema via llguidance: non-ASCII content and enum accepted, invented value / key rejected
  func testSchemaLlg() throws {
    let sch = #"{"type":"object","properties":{"art":{"type":"string","enum":["Übergang","Kampf"]},"was":{"type":"string"}},"required":["art","was"],"additionalProperties":false}"#
    func rej(_ t: String) throws -> Int? {
      let (g, _) = try Self.kit.gate(tag: sch, schema: true)
      defer { Self.kit.done(g, tag: sch) }
      for (i, id) in Self.tok.encode(text: t, addSpecialTokens: false).enumerated() { if try !g.allows(id) { return i }; try g.commit(id) }
      return nil
    }
    XCTAssertNil(try rej(#"{"art":"Übergang","was":"Räuber Müller schlich über die Brücke"}"#))
    XCTAssertNotNil(try rej(#"{"art":"Flucht""#))
    XCTAssertNotNil(try rej(#"{"wer":"x""#))
  }

  // MARK: response_format lark / regex (llguidance)
  static let lark = #"start: "ZIEL: " /[a-z]{1,5}/ "\n" "ANTWORT: " /[0-9]{1,3}/ "\n""#

  func larkReject(_ t: String, lang: String = "lark", src: String = GrammarTests.lark) throws -> Int? {
    let (g, _) = try Self.kit.gate(tag: src, lang: lang)
    defer { Self.kit.done(g, tag: src) }
    for (i, id) in Self.tok.encode(text: t, addSpecialTokens: false).enumerated() { if try !g.allows(id) { return i }; try g.commit(id) }
    return nil
  }

  func testLarkAcceptsAndRejects() throws {
    XCTAssertNil(try larkReject("ZIEL: abc\nANTWORT: 42\n"))
    XCTAssertNotNil(try larkReject("Hallo"))
    XCTAssertNotNil(try larkReject("ZIEL: abc\nANTWORT: x"))
    XCTAssertNil(try larkReject("2024-05", lang: "regex", src: "[0-9]{4}-[0-9]{2}"))
    XCTAssertNotNil(try larkReject("20x", lang: "regex", src: "[0-9]{4}-[0-9]{2}"))
  }

  func testLarkInvalidIsBadRequest() {
    XCTAssertThrowsError(try Self.kit.gate(tag: "start: nope(", lang: "lark")) { e in
      XCTAssertTrue(e is BadRequest, "\(e)"); XCTAssertTrue("\(e)".contains("llguidance"))
    }
    XCTAssertThrowsError(try Self.kit.gate(tag: "(", lang: "regex")) { XCTAssertTrue($0 is BadRequest) }
  }

  /// accepting state precedes EOS: EOS allowed, nothing else; after EOS is consumed the gate is inert (no crash, logits untouched)
  func testLarkStopAfterEos() throws {
    let src = #"start: "ab""#
    let (g, _) = try Self.kit.gate(tag: src, lang: "lark")
    defer { Self.kit.done(g, tag: src) }
    for id in Self.tok.encode(text: "ab", addSpecialTokens: false) { try g.commit(id) }
    XCTAssertTrue(try g.allows(Self.kit.eos))
    XCTAssertFalse(try g.allows(Self.tok.encode(text: "a", addSpecialTokens: false)[0]))
    let l = MLXArray(Array(repeating: Float(1), count: Self.kit.vocab))
    let m = try g.apply(l)
    XCTAssertEqual((m .> -Float.infinity).sum().item(Int.self), Self.kit.stop.count)
    try g.commit(Self.kit.eos)
    try g.commit(Self.kit.eos)   // one step past EOS
    XCTAssertTrue(try g.allows(7))
    XCTAssertEqual((try g.apply(l) .> -Float.infinity).sum().item(Int.self), Self.kit.vocab)
  }

  func testLarkAfterThink() throws {
    let end = Self.tok.convertTokenToId("</think>")!
    let (g, _) = try Self.kit.gate(tag: Self.lark, lang: "lark", after: end)
    defer { Self.kit.done(g, tag: Self.lark) }
    let free = Self.tok.encode(text: "Überlege frei 123", addSpecialTokens: false)
    for id in free { XCTAssertTrue(try g.allows(id)); try g.commit(id) }
    try g.commit(end)
    XCTAssertEqual(g.committed, 0)
    XCTAssertFalse(try g.allows(Self.tok.encode(text: "Hallo", addSpecialTokens: false)[0]))
    for id in Self.tok.encode(text: "ZIEL: ab\nANTWORT: 7\n", addSpecialTokens: false) { XCTAssertTrue(try g.allows(id)); try g.commit(id) }
  }

  func testVocab() {
    XCTAssertEqual(Self.kit.vocab, 150144)
    XCTAssertEqual(Self.tok.convertTokenToId("<think>"), 150006)
    XCTAssertEqual(Self.tok.convertTokenToId("<tool_call>"), 150010)
    XCTAssertEqual(Self.tok.convertTokenToId("<|observation|>"), 150005)
  }

  func testValidCallAccepted() throws {
    for t in ["<tool_call>get_weather<arg_key>city</arg_key><arg_value>Berlin</arg_value></tool_call>",
      "<tool_call>add<arg_key>a</arg_key><arg_value>2</arg_value><arg_key>b</arg_key><arg_value>-17</arg_value></tool_call>",
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Köln</arg_value><arg_key>unit</arg_key><arg_value>c</arg_value></tool_call>"] {
      let r = try firstReject(t)
      XCTAssertNil(r, "\(t) rejected at token \(r ?? -1): \(r.map { Self.tok.decode(tokenIds: Array(Self.tok.encode(text: t, addSpecialTokens: false).prefix($0 + 1))) } ?? "")")
    }
  }

  func testInventedArgRejected() throws {
    let t = "<tool_call>get_weather<arg_key>citty</arg_key>"
    let r = try firstReject(t)
    XCTAssertNotNil(r)
    XCTAssertNotNil(try firstReject("<tool_call>nope<arg_key>city</arg_key>"))
    XCTAssertNotNil(try firstReject("<tool_call>add<arg_key>a</arg_key><arg_value>\"x\"</arg_value>"), "string for integer")
  }

  func testRequiredArgMissingRejected() throws {
    XCTAssertNotNil(try firstReject("<tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"))
  }

  func testUmlautFreeText() throws {
    XCTAssertNil(try firstReject("Schöne Grüße aus Köln! Größe: 42 m² — „Äpfel“ ✓ "))
    XCTAssertNil(try firstReject("Ich rufe das Werkzeug auf. <tool_call>get_weather<arg_key>city</arg_key><arg_value>Zürich</arg_value></tool_call>"))
  }

  func testThinkOptional() throws {
    XCTAssertNil(try firstReject("<think>Überlege: Köln oder Zürich?</think>Antwort"))
    XCTAssertNil(try firstReject("Antwort ohne think"))
    XCTAssertNil(try firstReject("<think>x</think><tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value></tool_call>"))
  }

  func testStopTokensAllowedAfterCallAndText() throws {
    let (g, _) = try Self.kit.gate(tag: Self.tag)
    defer { Self.kit.done(g, tag: Self.tag) }
    for id in Self.tok.encode(text: "<tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value></tool_call>", addSpecialTokens: false) { try g.commit(id) }
    for s in [150001, 150003, 150005] { XCTAssertTrue(try g.allows(s), "stop \(s)") }
    let (g2, _) = try Self.kit.gate(tag: Self.tag)
    defer { Self.kit.done(g2, tag: Self.tag) }
    for id in Self.tok.encode(text: "<tool_call>add<arg_key>a</arg_key>", addSpecialTokens: false) { try g2.commit(id) }
    XCTAssertFalse(try g2.allows(150001), "no EOS mid-call")
  }

  func testNoToolsGrammarIsFreeTextPlusThink() throws {
    let tag = try GLM47().structuralTag(tools: [])
    XCTAssertNil(try firstReject("<think>a</think>Hallo Welt äöü", tag: tag))
    XCTAssertNotNil(try firstReject("<tool_call>x", tag: tag))
  }
}

final class ParserTests: XCTestCase {
  func parse(_ chunks: [String]) -> [Event] {
    let p = GLM47().parser(tools: tools)
    var ev: [Event] = []
    for c in chunks { ev += p.feed(c) }
    ev += p.finish()
    return ev
  }
  func merge(_ ev: [Event]) -> [Event] {
    var out: [Event] = []
    for e in ev {
      switch (out.last, e) {
      case (.reasoning(let a)?, .reasoning(let b)): out[out.count - 1] = .reasoning(a + b)
      case (.content(let a)?, .content(let b)): out[out.count - 1] = .content(a + b)
      default: out.append(e)
      }
    }
    return out
  }
  func testFull() {
    let s = "<think>denke</think>Ich suche.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Köln</arg_value></tool_call>"
    for n in [1, 3, 7, 1000] {   // arbitrary chunk boundaries
      let chunks = stride(from: 0, to: s.count, by: n).map { String(s.dropFirst($0).prefix(n)) }
      XCTAssertEqual(merge(parse(chunks)), [.reasoning("denke"), .content("Ich suche."), .call(name: "get_weather", args: #"{"city":"Köln"}"#)], "chunk \(n)")
    }
  }
  func testNoThinkAndTypedArgs() {
    XCTAssertEqual(merge(parse(["Hallo<tool_call>add<arg_key>a</arg_key><arg_value>2</arg_value><arg_key>b</arg_key><arg_value>3</arg_value></tool_call>"])),
      [.content("Hallo"), .call(name: "add", args: #"{"a":2,"b":3}"#)])
  }
}

final class RequiredTests: XCTestCase {
  func testRequiredForcesCall() throws {
    let kit = GrammarTests.kit!, tag = try GLM47().structuralTag(tools: tools, GrammarOpts(required: true))
    func rej(_ t: String) throws -> Int? {
      let (g, _) = try kit.gate(tag: tag); defer { kit.done(g, tag: tag) }
      for (i, id) in GrammarTests.tok.encode(text: t, addSpecialTokens: false).enumerated() { if try !g.allows(id) { return i }; try g.commit(id) }
      return nil
    }
    XCTAssertNil(try rej("<think>ok</think><tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value></tool_call>"))
    XCTAssertNil(try rej("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Bonn</arg_value></tool_call>"))
    XCTAssertNotNil(try rej("Hallo, kein Aufruf"))
  }
}

final class SchemaTests: XCTestCase {
  let tl: [JSON] = [["type": "function", "function": ["name": "edit", "parameters": [
    "type": "object",
    "properties": [
      "path": ["type": ["string", "null"]], "old": ["anyOf": [["type": "integer"], ["type": "string"]]], "any": [:] as JSON,
      "mode": ["type": "string", "enum": ["größe", "b"]], "n": ["type": "integer"], "flag": ["type": "boolean"],
    ] as JSON, "required": ["path", "old", "any", "mode"]] as JSON] as JSON]]

  func reject(_ t: String) throws -> Int? {
    let kit = GrammarTests.kit!, tag = try GLM47().structuralTag(tools: tl)
    let (g, _) = try kit.gate(tag: tag); defer { kit.done(g, tag: tag) }
    for (i, id) in GrammarTests.tok.encode(text: t, addSpecialTokens: false).enumerated() { if try !g.allows(id) { return i }; try g.commit(id) }
    return nil
  }
  func call(_ args: String) -> String { "<tool_call>edit" + args + "</tool_call>" }
  func a(_ k: String, _ v: String) -> String { "<arg_key>\(k)</arg_key><arg_value>\(v)</arg_value>" }

  func testStringishTypesTakeRawText() throws {
    // sorted order: any, mode, old, path (required), then optional
    let ok = call(a("any", "she said \"hi\"\n  x") + a("mode", "größe") + a("old", "foo bar") + a("path", "src/ä.swift") + a("n", "3") + a("flag", "true"))
    XCTAssertNil(try reject(ok))
  }
  func testNonStringStillJSON() throws {
    XCTAssertNotNil(try reject(call(a("any", "x") + a("mode", "b") + a("old", "y") + a("path", "p") + a("n", "\"three\""))))
    XCTAssertNotNil(try reject(call(a("any", "x") + a("mode", "nope"))))
  }
  func testParserKeepsStringsExactAndLastDuplicateWins() {
    let p = GLM47().parser(tools: tl)
    let txt = call(a("path", "  keep\n  ") + a("old", " 7 ") + a("any", "\tx") + a("n", " 5 ") + a("n", "6"))
    let ev = p.feed(txt) + p.finish()
    guard case .call(_, let args)? = ev.first else { return XCTFail("\(ev)") }
    let o = try! JSONSerialization.jsonObject(with: Data(args.utf8)) as! JSON
    XCTAssertEqual(o["path"] as? String, "  keep\n  ")
    XCTAssertEqual(o["old"] as? String, " 7 ")
    XCTAssertEqual(o["any"] as? String, "\tx")
    XCTAssertEqual(o["n"] as? Int, 6)
  }
  func testArgsParsedSequentially() {
    // a value that contains the closing tag text of a later arg must not confuse pairing
    let p = GLM47().parser(tools: tl)
    let ev = p.feed(call(a("path", "a") + a("old", "b"))) + p.finish()
    guard case .call(_, let args)? = ev.first else { return XCTFail() }
    XCTAssertEqual(args, #"{"path":"a","old":"b"}"#)
  }
}
