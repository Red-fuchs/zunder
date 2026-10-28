# Phase 2: real flint-int4 (2026-10-28)

Model: RecursiveSI/flint-int4 snapshot 3b93f905, HF_HUB_OFFLINE=1. M1 Max, gpu-run `--ram 26` (user-approved; `--ram 30` never passed the +6 GB gate).

## 1. Load + generate
Worked first try, no code fixes needed in the model port (per-tensor quant inference incl. 8-bit embeddings handled by Quant.swift). Load 4.3-7.5 s (page cache warm). Chat template prompt ids identical to python tokenizer on all 4 prompts.

## 2. Parity vs mlx-lm (greedy, 4 prompts, 64 tokens, bf16, `zunder gen` vs `ref/gen_ref.py`, compare: `ref/compare.py`)
| prompt | tokens | top-1 match | first divergence | max/mean abs dlogit (top-1) |
|---|---|---|---|---|
| capital of France | 27/27 (eos) | 27/27 | none | 0.64 / 0.15 |
| python palindrome | 64 | 64/64 | none | 0.57 / 0.09 |
| sky is blue | 64 | 64/64 | none | 0.91 / 0.10 |
| translate DE | 64 | 36/37 up to divergence | token 36 | 0.24 / 0.08 |

Deviation: token 36 of prompt 4 is a near-tie (zunder 260 @ 0.047 vs 15098 @ -0.025; ref 15098 @ 0.135 vs 260 @ -0.063), gap < 0.2 in both. Consistent with bf16 accumulation-order noise (logit deltas up to ~0.9 at logit magnitudes ~20+), not a port bug. Tiny-model logit parity test (swift test) already covered sliding-window wrap.

## 3. Bench (greedy, repeated-sentence prompt, 128 gen tokens, warmed; zunder after kernel warm-up)
| | zunder | mlx-lm ref |
|---|---|---|
| prefill 2048 tok | 279 tok/s | 241 tok/s |
| decode @2048 ctx | 42.5 tok/s | 37.4 tok/s |
| prefill 512 tok | 225 tok/s | 158 tok/s |
| decode @512 ctx | 39.3 tok/s | 41.4 tok/s |
| peak RAM | 23.9 GB footprint (16.6 GB max RSS, rest mmap'd weights) | 23.4-23.5 GB (mx peak) |

Grammar-constrained decode: mask 36-41 us/token (~0.1 %), decode speed same as unconstrained (43.0 tok/s @2048). Note: first generate after load is slow (kernel compile, prefill 12 tok took 7 s); `zunder bench` now does a warm-up run first (an un-warmed first bench read 20.8 tok/s, misleading). Short-prompt chat decode 40-43 tok/s on both.

## 4. Tool calls (server, `scripts/toolcall_test.py`; 4 tools, 7 prompts x {grammar on, `grammar:none`}; incl. tool_choice=required, German prompt, no-tool prompt)
Result 14/14 OK (7 prompts x 2 modes); all calls produced valid OpenAI `tool_calls` (name + JSON args with required keys, typed numbers/booleans, enum honored), `finish_reason=tool_calls`, `reasoning_content` separated. Unconstrained Flint was also 100 % valid on these easy prompts, so grammar adds a guarantee rather than a visible gain here; harder cases (long schemas, nested args) not tested.

## 5. fp16
Not done (not cheap: needs dtype cast path + re-parity). DFlash2: later.

## Open
- Reference loads tokenizer with a Mistral-regex warning (`fix_mistral_regex`); prompt ids still matched.
- Parity only on 4 short prompts; no long-context (>513 window) real-model parity yet (tiny-model covers it).
- First-token latency after load (kernel warm-up ~7 s).
