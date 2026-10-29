import time,mlx.core as mx,os
from mlx_lm import load
from huggingface_hub import snapshot_download
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
L=model.layers
print(model.lm_head if hasattr(model,'lm_head') else None)
a=L[3].self_attn; mo=L[3].mlp
print(a.q_proj,a.o_proj,mo.shared_expert.gate_proj,mo.gate,sep="\n")
def bench(f,n=6):
    r=[]
    for _ in range(n):
        t=time.perf_counter();mx.eval([f() for _ in range(20)]);r.append((time.perf_counter()-t)/20*1000)
    return min(r)
for M in (1,5,8):
    x=mx.random.normal((1,M,2048)).astype(mx.bfloat16)
    ao=mx.random.normal((1,M,4096)).astype(mx.bfloat16)
    print(M,"q",f"{bench(lambda:a.q_proj(x)):.3f}","k",f"{bench(lambda:a.k_proj(x)):.3f}","o",f"{bench(lambda:a.o_proj(ao)):.3f}",
      "shared",f"{bench(lambda:mo.shared_expert(x)):.3f}","gate",f"{bench(lambda:mo.gate(x.astype(mx.float32))):.3f}","lmhead",f"{bench(lambda:model.lm_head(x)):.3f}",flush=True)
