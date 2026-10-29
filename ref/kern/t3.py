import time,mlx.core as mx
from mlx_lm import load
from huggingface_hub import snapshot_download
import dense_kern as D
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
L=model.layers; a=L[3].self_attn; mo=L[3].mlp
def bench(f,n=12):
    r=[]
    for _ in range(n):
        t=time.perf_counter();mx.eval([f() for _ in range(20)]);r.append((time.perf_counter()-t)/20*1000)
    return min(r)
for name,ql,K in (("lm_head",model.lm_head,2048),("q",a.q_proj,2048),("o",a.o_proj,4096)):
  for M in (5,8):
    for CH in (4,8):
      x=mx.random.normal((1,M,ql.weight.shape[1]*8)).astype(mx.bfloat16)
      ref=ql(x);new=D.qmv(x,ql,CH);mx.eval(ref,new)
      err=(mx.abs(ref.astype(mx.float32)-new.astype(mx.float32)).max()/mx.abs(ref.astype(mx.float32)).max()).item()
      print(name,M,CH,f"nat {bench(lambda:ql(x)):.3f} new {bench(lambda:D.qmv(x,ql,CH)):.3f} relerr {err:.4f}",flush=True)
