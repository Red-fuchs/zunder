# zunder

Local Swift MLX inference server for agents. OpenAI-compatible (`/v1/chat/completions` stream + non-stream, `/v1/models`, `/health`),
127.0.0.1 only, one model, one request at a time. Tool calls are constrained per token by xgrammar structural tags
(vendored in mlx-swift-lm); reasoning stays free (`reasoning_content`). Adds the `flint` model type (RecursiveSI/flint) to mlx-swift-lm 3.32.3.

    ./build.sh                                    # needs Xcode (Metal shaders); keep mlx-swift_Cmlx.bundle next to the binary
    zunder start --model RecursiveSI/flint-int4   # local dir or HF-cache id, never downloads; port via `portwart claim zunder`
    zunder status | log [-f] | stop
    zunder bench --model <m> [--tokens 128] [--prompt-tokens 512]
    gpu-run -n "zunder bench" --ram 28 zunder bench --model RecursiveSI/flint-int4 --no-lease   # big models: wrap in gpu-run

- GPU slot: the model is loaded only while a `gpu-run` lease is held (holder job, FIFO + RAM gate); `--no-lease` for tiny/test models, `--ram GB` overrides the estimate.
- Idle unload after `--idle` s (default 600): weights freed, slot released, next request reloads (re-queues).
- Defaults temp 1.0 / top_p 0.95 / top_k 20 (request fields override).
- Request extras: `"grammar":"none"` (no constraint), `"think":false` (grammar bans `<think>`), `tool_choice` auto|none|required|{function}, `parallel_tool_calls:false` (with required).
- Tool formats: `ToolFormat` protocol, `GLM47` for flint/GLM4 (`Sources/ZunderCore/ToolFormat.swift`); Qwen/Gemma/Hermes later.

### Raw grammar (`response_format` lark / regex)

llguidance constrains the whole output (or only the answer after the think block). Not combinable with `tools` (400); invalid grammar -> 400 with the llguidance error.
`scope`: `all` (default) or `after_think` (native think block stays free, grammar starts after `</think>`; no effect with `"think":false`). `"type":"regex","regex":"..."` works the same way. Compiled grammars are cached.

    curl -s localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d "$(jq -n --rawfile g ~/Claude-Workspace/think-grammar/skeleton.lark \
      '{messages:[{role:"user",content:"Wie viel ist 17*23?"}],max_tokens:1500,response_format:{type:"lark",lark:$g,scope:"all"}}')"

Tests: `swift test` (needs `fixtures/tiny-flint` = tiny-random/flint, `fixtures/flint-tok` = flint-int4 tokenizer/config files, and
`ref/out/tiny.safetensors` from `ref/dump_logits.py` in a venv with mlx-lm main). Smoke: `scripts/smoke.sh` (server + grammar-forced tool call on tiny-random/flint).
