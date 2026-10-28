import sys, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
path, np_, ng = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model, tok = load(path)
ids = tok.encode("The quick brown fox jumps over the lazy dog. " * (np_ // 8 + 1), add_special_tokens=False)[:np_]
cache = make_prompt_cache(model)
t0 = time.time()
for i in range(0, len(ids), 512):
    lg = model(mx.array([ids[i:i+512]]), cache=cache); mx.eval(lg)
pre = time.time() - t0
y = mx.argmax(lg[0, -1]); t1 = time.time()
for _ in range(ng):
    lg = model(y.reshape(1, 1), cache=cache); y = mx.argmax(lg[0, -1]); mx.eval(y)
dec = time.time() - t1
print(f"reference: prefill {len(ids)} tok {pre:.2f}s ({len(ids)/pre:.0f} tok/s) gen {ng} tok {dec:.2f}s ({ng/dec:.1f} tok/s) peak {mx.get_peak_memory()/1e9:.1f} GB")
