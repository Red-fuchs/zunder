import sys, json, time, urllib.request
B = sys.argv[1]
T = lambda n, d, props, req: {"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": props, "required": req}}}
tools = [
 T("get_weather", "Get current weather for a city", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
 T("add", "Add two numbers", {"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"]),
 T("send_email", "Send an email", {"to": {"type": "string"}, "subject": {"type": "string"}, "urgent": {"type": "boolean"}}, ["to", "subject"]),
 T("search_files", "Search files by glob", {"pattern": {"type": "string"}, "max_results": {"type": "integer"}}, ["pattern"]),
]
cases = [("What's the weather in Berlin in celsius?", "get_weather", "auto"), ("Compute 17.5 plus 24.25.", "add", "auto"),
 ("Mail bob@example.com with subject 'Lunch' and mark it urgent.", "send_email", "auto"), ("Find all python files, at most 5.", "search_files", "auto"),
 ("Wie ist das Wetter in München?", "get_weather", "required"), ("Add 3 and 4", "add", "required"), ("Say hello.", None, "auto")]
ok = 0
for q, want, tc in cases:
    for gram in (None, "none"):
        body = {"messages": [{"role": "user", "content": q}], "tools": tools, "tool_choice": tc, "max_tokens": 1500}
        if gram: body["grammar"] = gram
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=900))
        m = r["choices"][0]["message"]; calls = m.get("tool_calls") or []
        good = False
        try:
            if want is None: good = True
            else:
                c = calls[0]["function"]; a = json.loads(c["arguments"]); good = c["name"] == want and all(k in a for k in next(x for x in tools if x["function"]["name"] == want)["function"]["parameters"]["required"])
        except Exception as e: good = False
        ok += good
        print(f"[{'constrained' if not gram else 'free'}] {q[:45]!r} want={want} tc={tc} -> {'OK' if good else 'FAIL'} calls={json.dumps([(c['function']['name'], c['function']['arguments']) for c in calls], ensure_ascii=False)} finish={r['choices'][0]['finish_reason']} reasoning_len={len(m.get('reasoning_content') or '')} content={(m.get('content') or '')[:60]!r} {time.time()-t:.1f}s", flush=True)
print(f"{ok}/{len(cases)*2}")
