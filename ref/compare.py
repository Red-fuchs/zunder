import sys, json, numpy as np
z, r = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for a, b in zip(z, r):
    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb)); d = next((i for i in range(n) if ta[i] != tb[i]), None)
    # logits on the shared prefix before first divergence
    k = n if d is None else d + 1
    err = [max(abs(x - y) for x, y in zip(sa["logits"][:1], sb["logits"][:1])) for sa, sb in zip(a["top8"][:k], b["top8"][:k])]
    top1 = sum(sa["ids"][0] == sb["ids"][0] for sa, sb in zip(a["top8"][:k], b["top8"][:k]))
    print(f"{a['prompt'][:40]!r}: prompt_ids_equal={a['prompt_ids']==b['prompt_ids']} len z/r={len(ta)}/{len(tb)} first_diverge={d} top1_match={top1}/{k} max|dlogit_top1|={max(err):.3f} mean={np.mean(err):.3f}")
