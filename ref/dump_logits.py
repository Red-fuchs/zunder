import sys, numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
path, out = sys.argv[1], sys.argv[2]
model, _ = load(path)
rng = np.random.RandomState(0)
ids = rng.randint(0, 150000, size=700).tolist()   # > sliding window (513): exercises rotating cache + window mask
nxt = rng.randint(0, 150000, size=4).tolist()
cache = make_prompt_cache(model)
lg = model(mx.array([ids]), cache=cache).astype(mx.float32)
res = {"ids": np.array(ids), "nxt": np.array(nxt), "prefill_last": np.array(lg[0, -1]), "prefill_head": np.array(lg[0, :4])}
for i, t in enumerate(nxt):
    lg = model(mx.array([[t]]), cache=cache).astype(mx.float32)
    res[f"step{i}"] = np.array(lg[0, -1])
mx.save_safetensors(out, {k: mx.array(v.astype(np.int32) if v.dtype.kind=="i" else v) for k, v in res.items()})
print({k: v.shape for k, v in res.items()})
