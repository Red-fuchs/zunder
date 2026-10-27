#!/bin/sh
# Server smoke test on tiny-random/flint (random weights): grammar-forced tool call, non-stream + stream. Light, no gpu-run needed.
set -e
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
CFG=${CFG:-debug}
swift build -c "$CFG" >/dev/null
BIN=.build/$CFG/zunder
MODEL=${MODEL:-fixtures/tiny-flint}
$BIN start --model "$MODEL" --no-lease --idle 120
trap '$BIN stop' EXIT
P=$(cat ~/.local/state/zunder/port)
B=http://127.0.0.1:$P
curl -sf $B/health; echo
curl -sf $B/v1/models; echo
REQ='{"model":"tiny","messages":[{"role":"user","content":"Stelle den Modus auf a."}],"max_tokens":200,"temperature":1.0,
"tools":[{"type":"function","function":{"name":"set_mode","description":"set mode","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["a","b"]}},"required":["mode"]}}}],
"tool_choice":"required","think":false%s}'
echo "--- non-stream"
curl -sf $B/v1/chat/completions -H 'content-type: application/json' -d "$(printf "$REQ" '')" | tee /tmp/claude-501/zunder-smoke.json | python3 -c '
import sys,json
r=json.load(sys.stdin); m=r["choices"][0]["message"]; print(json.dumps(m,ensure_ascii=False)); print("finish:",r["choices"][0]["finish_reason"])
tc=m["tool_calls"][0]["function"]; assert tc["name"]=="set_mode" and json.loads(tc["arguments"])["mode"] in ("a","b"); print("OK non-stream tool call")'
echo "--- stream"
curl -sfN $B/v1/chat/completions -H 'content-type: application/json' -d "$(printf "$REQ" ',"stream":true')" | python3 -c '
import sys,json
calls=[];fin=None
for l in sys.stdin:
    if not l.startswith("data: ") or l.startswith("data: [DONE]"): continue
    d=json.loads(l[6:]); c=d["choices"][0]
    calls+=c["delta"].get("tool_calls",[]); fin=c["finish_reason"] or fin
print(calls, fin); assert calls and fin=="tool_calls"; print("OK stream tool call")'
tail -n 4 ~/.local/state/zunder/zunder.log
