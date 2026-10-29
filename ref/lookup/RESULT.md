# Flint prompt-lookup (n-gram 4..2, greedy, M1 Max, mlx #4787)

```
1 rewrite-code K=3 plain=39.5 lookup=36.9 x0.94 tok/step=1.26 acc/draftstep=0.97 hit=0.27 newtok=350 parity=NO@105
1 rewrite-code K=5 plain=39.5 lookup=35.5 x0.90 tok/step=1.43 acc/draftstep=1.67 hit=0.26 newtok=350 parity=NO@71
1 rewrite-code K=8 plain=39.5 lookup=31.2 x0.79 tok/step=1.39 acc/draftstep=1.57 hit=0.25 newtok=350 parity=NO@71
2 tool-call K=5 plain=41.3 lookup=35.5 x0.86 tok/step=1.37 acc/draftstep=1.27 hit=0.29 newtok=350 parity=NO@44
3 german-chat K=5 plain=42.5 lookup=37.6 x0.88 tok/step=1.07 acc/draftstep=0.72 hit=0.10 newtok=350 parity=NO@107```

Parity NO everywhere incl. 29-token prompt (no window wrap) -> likely S>1 vs S=1 quantized-kernel numeric near-tie flips, not rollback bug (unproven). Plain here ~40 tok/s (sync, python loop).
Verdict: net slowdown (0.79-0.94x); low hit rate (10-29%) and verify cost > gain.
