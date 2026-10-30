# Python decode trace, flint-int4 (2026-10-29, /tmp/trace.log)
| mode | ms/tok | tok/s |
|---|---|---|
| A pipelined (async eval) | 20.03 | 49.9 |
| B sync (build 3.11 + wait 19.99) | 23.10 | 43.3 |
| C chain8 | 22.36 | 44.7 |
| D compile | skipped | - |

GPU time ~20 ms/tok; graph build ~3 ms/tok is fully hidden by pipelining.
vs zunder (docs/FP16.md, short ctx): bf16 42.1, fp16 47.0 tok/s -> zunder ~6-16 % below pipelined ceiling.
dtype of the Python trace unknown; ctx length short. Tokenizer warning: flint-int4 needs fix_mistral_regex=True (tokenization correctness).
