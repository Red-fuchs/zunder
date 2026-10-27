import sys, numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
path, out = sys.argv[1], sys.argv[2]
model, _ = load(path)
rng = np.random.RandomState(0)
ids = rng.randint(0, 150000, size=700).tolist()   # > sliding window (513)
nxt = rng.randint(0, 150000, size=4).tolist()
wtok = rng.randint(0, 150000, size=620).tolist()
f32 = lambda x: np.array(x.astype(mx.float32))
res = {"ids": np.array(ids), "nxt": np.array(nxt), "wtok": np.array(wtok)}

# A: single-shot prefill + 4 decode steps
cache = make_prompt_cache(model)
lg = model(mx.array([ids]), cache=cache)
res["prefill_last"] = f32(lg[0, -1]); res["prefill_head"] = f32(lg[0, :4])
for i, t in enumerate(nxt):
    res[f"step{i}"] = f32(model(mx.array([[t]]), cache=cache)[0, -1])

# B: chunked prefill 512 + 188, then 4 decode steps
cache = make_prompt_cache(model)
model(mx.array([ids[:512]]), cache=cache)
lg = model(mx.array([ids[512:]]), cache=cache)
res["chunk_last"] = f32(lg[0, -1])
for i, t in enumerate(nxt):
    res[f"cstep{i}"] = f32(model(mx.array([[t]]), cache=cache)[0, -1])

# C: teacher-forced decode far past the window wrap (700 + 620 > 2 * 513)
cache = make_prompt_cache(model)
model(mx.array([ids]), cache=cache)
for i, t in enumerate(wtok):
    lg = model(mx.array([[t]]), cache=cache)
    if i % 100 == 99 or i == len(wtok) - 1: res[f"w{i}"] = f32(lg[0, -1])
mx.save_safetensors(out, {k: mx.array(v.astype(np.int32) if v.dtype.kind == "i" else v) for k, v in res.items()})
print(sorted(res)[:6], len(res))
