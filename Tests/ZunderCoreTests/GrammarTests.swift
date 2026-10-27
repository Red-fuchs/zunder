import XCTest
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
      kit = try GrammarKit(tok: tok, eos: tok.convertTokenToId("<|end_of_text|>")!, stop: stop)
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
