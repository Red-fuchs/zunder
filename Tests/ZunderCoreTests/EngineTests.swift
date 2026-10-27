import XCTest
@testable import ZunderCore

final class EngineTests: XCTestCase {
  func eng() async throws -> Engine { try await Engine(spec: root.appendingPathComponent("fixtures/tiny-flint").path) }

  func testTemplate() async throws {
    let e = try await eng()
    let msgs: [JSON] = [
      ["role": "user", "content": "Wetter in Köln?"],
      ["role": "assistant", "reasoning_content": "Ich muss nachsehen.", "content": "", "tool_calls": [["id": "c1", "type": "function", "function": ["name": "get_weather", "arguments": ["city": "Köln"] as JSON] as JSON] as JSON]],
      ["role": "tool", "tool_call_id": "c1", "content": "12 Grad"],
    ]
    let ids = try e.prompt(messages: msgs, tools: tools)
    let s = e.tok.decode(tokenIds: ids, skipSpecialTokens: false)
    print(s)
    XCTAssertTrue(s.hasSuffix("<|assistant|>"), "generation prompt")
    XCTAssertTrue(s.contains("<think>Ich muss nachsehen.</think>"))
    XCTAssertTrue(s.contains("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Köln</arg_value></tool_call>"))
    XCTAssertTrue(s.contains("<|observation|><tool_response>12 Grad</tool_response>"))
    XCTAssertTrue(s.contains("\"name\":\"get_weather\"") || s.contains("\"name\": \"get_weather\""))
  }

  func testDetokKeepsMarkers() async throws {
    let e = try await eng()
    var d = Detok(e.tok)
    var s = ""
    for t in [150006] + e.tok.encode(text: "Grüße", addSpecialTokens: false) + [150007] { s += d.push(t) }
    XCTAssertEqual(s, "<think>Grüße</think>")
  }

  func testGenerateWithGrammarOnTiny() async throws {
    let e = try await eng()
    let tool: JSON = ["type": "function", "function": ["name": "set_mode", "parameters": ["type": "object", "properties": ["mode": ["type": "string", "enum": ["a", "b"]]], "required": ["mode"]] as JSON] as JSON]
    let ids = try e.prompt(messages: [["role": "user", "content": "x"]], tools: [tool])
    let tag = try e.fmt.structuralTag(tools: [tool], GrammarOpts(required: true, think: false, parallel: false))
    let out = try e.generate(ids: ids, tag: tag, params: SampleParams(), maxTokens: 100) { _ in true }
    var d = Detok(e.tok); let p = e.fmt.parser(tools: [tool])
    var ev: [Event] = []
    for t in out.tokens { ev += p.feed(d.push(t)) }
    ev += p.finish()
    print(out.stats.line, ev)
    XCTAssertEqual(out.finish, "stop")
    XCTAssertTrue(ev.contains { if case .call(let n, _) = $0 { return n == "set_mode" }; return false })
    XCTAssertFalse(ev.contains { if case .content = $0 { return true }; return false })
  }
}
