# Qwen3.6-35B-A3B JANGTQ4 vs Flint-int4, M1 Max, greedy, 256 tok (2026-10-29)
JANG loaded via /opt/miniconda3 python (mlx 0.31.2 + jang_tools load_jangtq_model); Flint via zunder-4787 (mlx 0.32.5dev). Different mlx builds. Prefill tps = 512-chunk, includes first-chunk cost (short = noisy).

| ctx | Qwen3.6 decode tok/s (ms) | Flint decode tok/s (ms) | Qwen prefill | Flint prefill |
|---|---|---|---|---|
| ~25 | 33.6 (29.8) | 45.7 (21.9) | 68 | 134 |
| 2k | 32.6 (30.7) | 42.3 (23.7) | 110 | 214 |
| 8k | 29.4 (34.1) | 37.4 (26.8) | 109 | 217 |

Verify forward (ms, 2k cache): M=1/2/4/8: Qwen 30.7/45.9/82.3/142.8 ; Flint 26.3/39.9/59.9/100.0
(verify uses cache trim; Qwen3.6 has linear-attn layers, trim may not rewind their state - cost only, not correctness).
Qwen sample (German, think-tags empty): "Gerne! Hier ist die Erklärung ... Warum ist der Himmel blau? ..." sane.
