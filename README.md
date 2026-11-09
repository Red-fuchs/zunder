# zunder

**A local LLM server in Swift + MLX that makes small models reliable for agents.**

zunder runs on Apple Silicon, speaks the OpenAI API, and constrains tool calls and structured output **per token** with a grammar.
A 4B or MoE model on a laptop then emits valid `tool_calls` every time. No cloud, no data leaving the machine.

```
agent / app ──OpenAI API──▶ zunder (127.0.0.1) ──▶ MLX on Metal
                               │
                               ├─ xgrammar structural tags → valid tool calls
                               ├─ llguidance → JSON schema / lark / regex output
                               └─ free reasoning block (`reasoning_content`)
```

## Why

Small local models are cheap, private and fast enough, but they sometimes break JSON or tool-call syntax.
That breaks an agent loop. zunder removes the failure mode at decode time: the sampler can only pick tokens the grammar allows.
The model still thinks freely in its reasoning block.

## Features

- **OpenAI-compatible**: `/v1/chat/completions` (stream + non-stream), `/v1/models`, `/health`
- **Grammar-constrained tool calls**: `tool_choice` auto | none | required | {function}, typed arguments from the JSON schema
- **Structured output**: `response_format` json_schema, lark or regex via llguidance. Scope `all` or `after_think` (the reasoning block stays free)
- **Thinking control**: `"think": false` bans the think block via grammar
- **Models**: Flint MoE (own Swift port of the mlx-lm model) plus the model families of mlx-swift-lm, e.g. Gemma 4; mixed-quant (JANG) configs
- **fp16 compute on M1/M2**, where bf16 is emulated: `--dtype auto`
- **Shared-machine friendly**: loads weights only while it holds a GPU lease from a local FIFO queue (`--no-lease` to skip), unloads after idle (default 10 min)
- **CLI**: `zunder start | stop | status | log | bench`

## Numbers (M1 Max 64 GB, Flint int4 MoE, warmed)

| | zunder (Swift) | mlx-lm (Python) |
|---|---|---|
| prefill 2048 tok | 279 tok/s | 241 tok/s |
| prefill 512 tok | 225 tok/s | 158 tok/s |
| decode @2048 ctx | 42.5 tok/s | 37.4 tok/s |
| decode @512 ctx | 39.3 tok/s | 41.4 tok/s |

- Grammar mask cost: **36–41 µs/token (~0.1 %)**. Constrained decode runs as fast as unconstrained.
- Tool-call test: **14/14 valid** `tool_calls` (7 prompts × grammar on/off, incl. `tool_choice=required` and German prompts).
- fp16 on M1: decode +4–12 %, prefill +10–12 % vs bf16.
- Logit parity with mlx-lm checked on a tiny random model and 4 real prompts.

Details: [docs/PHASE2.md](docs/PHASE2.md), [docs/FP16.md](docs/FP16.md). Single runs on a shared machine, noise ±3 tok/s.

## Quick start

```bash
./build.sh                                  # needs Xcode (Metal shaders)
zunder start --model <local dir or HF-cache id> --no-lease
curl -s localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Wie wird das Wetter in Sofia?"}],
  "tools":[{"type":"function","function":{"name":"wetter","parameters":{"type":"object",
    "properties":{"ort":{"type":"string"}},"required":["ort"]}}}],
  "tool_choice":"required"}'
```

zunder never downloads models. Point it at weights you already have. More flags and request extras: [docs/USAGE.md](docs/USAGE.md).

## Status

Personal research project, used daily as the LLM backend for my local agents (voice assistant, meeting-minutes pipeline).
One model and one request at a time, localhost only. Not a hardened multi-user server.
`ref/` holds Python reference scripts and benchmark experiments, including negative results.

## Credits

- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) and mlx-swift-lm (MIT, Apple)
- Grammar layout ported from [recursive-oss/rec](https://github.com/recursive-oss/rec) (Apache-2.0)
- [xgrammar](https://github.com/mlc-ai/xgrammar), [llguidance](https://github.com/guidance-ai/llguidance)

See [NOTICE](NOTICE).

## License

MIT, see [LICENSE](LICENSE). Ported parts keep their original licenses (see NOTICE).
