# zunder — local Swift MLX inference server (Flint first, multi-model)

Name: Flint → Zunder. OpenAI-compatible local server + CLI for agents (pi, OpenCode), grammar-constrained tool calls.

## Goals
- Serve any mlx-swift-lm text model (Gemma 4, Qwen3.x, GLM4 MoE, ...) plus **Flint** (RecursiveSI/flint-int4, model_type `flint`), which mlx-swift-lm lacks.
- Tool calls constrained per token by xgrammar structural tags (vendored in mlx-swift-lm, `MLXGuidedGeneration.GrammarConstraint(tokenizer:structuralTag:)`), reasoning left free.
- Lean: CLI + skill, no MCP. Shared M1 Max 64 GB: idle unload, gpu-run lease.

## Building blocks (verified 2026-10-27)
- mlx-swift-lm **3.32.3** (pin exact). Public `LLMTypeRegistry.shared.registerModelType("flint") { data in ... }` (actor, await) → no fork needed.
- `MLXFast.scaledDotProductAttention(..., sinks:)` exists; MLXLMCommon `attentionWithCacheUpdate` has NO sinks → call MLXFast directly in Flint attention.
- `RotatingKVCache` exists; trim bugs upstream #424/#618 (untrimmable after wrap) → matters for rollback/speculation later.
- GLM XML tool parser: `MLXLMCommon/Tool/Parsers/GLM4ToolCallParser.swift`, `ToolCallFormat.glm4`. Other formats in `ToolCallFormat` (qwen3_5, gemma4, xml_function, ...).
- xgrammar vendored v0.1.30. Known: 2-byte UTF-8 bug in const_string (sl: route non-ASCII via regex), bounded repeats explode compile time.

## Flint model (port from Python reference)
- Reference: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/flint.py (326 lines, merged PR #1998, 2026-10-24) + `mlx_lm/kernels/expert_routing.py` (use the pure-MLX routing branch first).
- 49 layers (2 dense, 47 MoE), hidden 2048, 32 Q heads / 4 KV heads, head_dim 128, 35 sliding-window layers (window 513, RotatingKVCache), 14 full-attention layers without positional encoding, per-head QK RMSNorm, learned attention sinks, 256 routed experts top-6 sigmoid renormalized + ungated shared expert, SwitchGLU. Check `layer_types` in config.json.
- Weights: flint-int4 = `mlx-affine-int4`, quantization `{"bits":4,"group_size":64}`, 5 shards, 22.6 GB. Sanitize like flint.py.
- Tokenizer special tokens (single ids): 150006 `<think>`, 150007 `</think>`, 150010 `<tool_call>`, 150011 `</tool_call>`, 150012/13 `<arg_key>`, 150014/15 `<arg_value>`, 150005 `<|observation|>`, 150003 `<|user|>`, 150004 `<|assistant|>`, eos `<|end_of_text|>`. Vocab 150144.
- Chat template: generation prompt ends at `<|assistant|>` WITHOUT `<think>` → replies may start as content (vLLM PR #63957 note: ~1/800 skip think). Parser must not assume reasoning-first.
- Sampling defaults: temp 1.0, top_p 0.95, top_k 20. Keep `reasoning_content` of past assistant turns in history.

## Tool-call grammar (port of rec, Apache-2.0, attribute in NOTICE)
- Source: https://raw.githubusercontent.com/recursive-oss/rec/HEAD/engine/src/rec_mlx/grammar_builder.py (93 lines) — `build_flint_glm_grammar`: `Sequence[Optional(<think>AnyText(excl </think>)</think>), AnyText(excludes protocol markers), Or(Optional(<|user|>), Sequence(glm_4_7 tool-call tag, <|observation|>))]`, stops `<|end_of_text|>`,`<|user|>`,`<|observation|>`. Tool params must be JSON-schema objects; arg values typed per schema (strings raw, others JSON).
- xgrammar 0.1.30 has no `get_model_structural_tag("glm_4_7")` helper (that's Python 0.2.x) → build the structural-tag JSON by hand in Swift (per tool: name const + arg_key/arg_value pairs from schema properties, required args present).
- Design for multi-model: `protocol ToolFormat { func structuralTag(tools:) -> String; func parse(...) }`, implementations: `GLM47` (Flint, GLM4), later `Qwen35`, `Gemma4`, `Hermes`. Model adapter picks format from config/model_type. Grammar off per request with `"grammar":"none"`.
- Parse calls → OpenAI `tool_calls` (non-stream + streaming deltas), `reasoning_content` separated.

## Server
- OpenAI `/v1/chat/completions` (stream + non-stream, tools, tool_choice auto/none), `/v1/models`, health. Bind 127.0.0.1 only; port via `portwart claim zunder`.
- One model loaded at a time, single request at a time (queue). Prompt KV cache reuse across turns if cheap (prefix match), otherwise skip in v1.
- fp16 compute option (`--dtype fp16`, M1: +15–25 % measured in Locals) — default bf16 until parity checked.
- CLI: `zunder serve --model <dir|hf-id> [--port] [--idle 600]`, `zunder start|stop|status|log` (background like gemma-serve), `zunder bench` (tok/s prefill/decode, mask µs/token).
- Idle unload ≤ 600 s; GPU slot via gpu-run lease (port `~/Projects/sl/Sources/SLCore/Lease.swift`).

## Reuse from ~/Projects/sl (read, don't depend on it)
- `LM.swift` sample loop with grammar mask (mask cost 0.25–0.7 ms/token = 1–3 % of a step), snapshot/restore of cache, matcher reuse via rollback.
- `Lease.swift` (gpu-run holder + keep file + pid watchdog). Add acquire timeout + release on SIGTERM.
- Use `LLMModelFactory` text path, NOT the VLM factory (+2 GB towers).

## Later (not v1)
- DFlash2 speculative decoding (incoai/Flint-DFlash2, 0.4B, block 5–8; M1 Max issue z-lab/dflash#160: w4a16 acceptance collapse).
- Small-M 4-bit g64 matmul kernels from Locals (~3× on verify8, `~/Claude-Workspace/locals-bench`, Obsidian note "Locals – MLX-Kernels, Batching & JANG").
- Mixed-bit per-layer (JANG) models.

## Testing without the big model
- `tiny-random/flint` (21.9 MB random-init, same arch) for load/forward/server smoke tests.
- Logit parity vs Python mlx-lm main on tiny-random/flint (venv with `mlx-lm @ git+https://github.com/ml-explore/mlx-lm@main`).
- Grammar/parser unit tests with real Flint tokenizer files only (tokenizer.json, tokenizer_config.json, config.json, chat_template.jinja from RecursiveSI/flint-int4).
- Real flint-int4 lands in HF cache tonight (launchd job 23:00, ~17 h). Phase 2 = parity + bench on it.
