# Flint prompt-lookup (n-gram 4..2, greedy, M1 Max, mlx #4787)

```
1 rewrite-code K=3 plain=39.5 lookup=36.9 x0.94 tok/step=1.26 acc/draftstep=0.97 hit=0.27 newtok=350 parity=NO@105
1 rewrite-code K=5 plain=39.5 lookup=35.5 x0.90 tok/step=1.43 acc/draftstep=1.67 hit=0.26 newtok=350 parity=NO@71
1 rewrite-code K=8 plain=39.5 lookup=31.2 x0.79 tok/step=1.39 acc/draftstep=1.57 hit=0.25 newtok=350 parity=NO@71
2 tool-call K=5 plain=41.3 lookup=35.5 x0.86 tok/step=1.37 acc/draftstep=1.27 hit=0.29 newtok=350 parity=NO@44
3 german-chat K=5 plain=42.5 lookup=37.6 x0.88 tok/step=1.07 acc/draftstep=0.72 hit=0.10 newtok=350 parity=NO@107```

Parity NO everywhere incl. 29-token prompt (no window wrap) -> likely S>1 vs S=1 quantized-kernel numeric near-tie flips, not rollback bug (unproven). Plain here ~40 tok/s (sync, python loop).
Verdict: net slowdown (0.79-0.94x); low hit rate (10-29%) and verify cost > gain.

## Round 2 (bench2.py): thinking + lossy top-k acceptance
Thinking: template has NO enable_thinking flag (only m.reasoning_content on history). Default gen prompt `<|assistant|>`; model emits `<think>...` itself in all 3 workloads (code/tool/german: "We need answer user...", "Deconstruct the User's Request"). Prefill `<think>\n\n</think>\n\n` does NOT stop reasoning: code and german still start with untagged reasoning prose ("We need rewrite complete file..."); only tool-call workload goes straight to JSON. So thinking is not truly off for 2/3 (code ast check FAIL = no code block within 350 tok).
```
workload K=5 topk plain lookup x tok/step hit match-vs-plain check
1 code    1  41.3 35.0 0.85 1.44 0.28 76% no code block in 350tok (still reasoning)
1 code    2  41.3 34.5 0.84 1.45 0.27  5%
1 code    4  41.3 55.5 1.35 2.80 0.43  5%
2 tool    1  42.1 35.9 0.85 1.99 0.58 100% json ok, args==expected
2 tool    2  42.1 61.4 1.46 3.71 0.75  2% json FAIL
2 tool    4  42.1 72.2 1.72 4.00 0.75  3% json ok, args WRONG
3 german  1  40.4 31.5 0.78 1.09 0.12 18%
3 german  2  40.4 35.2 0.87 1.16 0.10  9%
3 german  4  40.4 38.7 0.96 1.19 0.09  9%
```
Match = positional token equality (diverges early, so low % after first fork is expected). Verdict: exact lookup still a slowdown; lossy top-k speeds up (1.35-1.72x) but breaks tool JSON/args correctness -> not usable for tool calls.
