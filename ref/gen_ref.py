import sys, json, time, numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
path, pf, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
model, tok = load(path)
res = []
for p in json.load(open(pf)):
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict): ids = ids["input_ids"]
    cache = make_prompt_cache(model)
    t0 = time.time(); lg = model(mx.array([ids]), cache=cache)[0, -1].astype(mx.float32); mx.eval(lg); pre = time.time() - t0
    toks, top = [], []
    t1 = time.time()
    for _ in range(n):
        i = mx.argsort(-lg)[:8]; v = lg[i]
        t = int(i[0].item()); top.append({"ids": i.tolist(), "logits": v.tolist()})
        if t in (150001, 150003, 150005, 150018): break
        toks.append(t)
        lg = model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32); mx.eval(lg)
    dec = time.time() - t1
    res.append({"prompt": p, "prompt_ids": ids, "tokens": toks, "top8": top, "prefill_s": pre, "dec_s": dec, "stats": f"prefill {len(ids)} tok {pre:.2f}s ({len(ids)/pre:.0f} tok/s) gen {len(toks)} tok {dec:.2f}s ({len(toks)/dec:.1f} tok/s)"})
    print(res[-1]["stats"], "peak GB", mx.get_peak_memory() / 1e9, flush=True)
json.dump(res, open(out, "w"))
