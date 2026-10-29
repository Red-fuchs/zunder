import time,mlx.core as mx,numpy as np,sys
from mlx_lm import load
from huggingface_hub import snapshot_download
import moe_kern as K
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
L=model.layers if hasattr(model,'layers') else model.model.layers
moe=L[10].mlp; sw=moe.switch_mlp
print(sw.gate_proj.weight.dtype,sw.gate_proj.scales.dtype,sw.gate_proj.weight.shape,sw.down_proj.weight.shape)
def bench(f,n=20):
    for _ in range(3):mx.eval(f())
    r=[]
    for _ in range(n):
        t=time.perf_counter();o=[f() for _ in range(10)];mx.eval(o);r.append((time.perf_counter()-t)/10*1000)
    return min(r)
for Mt in (1,5,8):
    x=(mx.random.normal((1,Mt,2048))*0.5).astype(mx.bfloat16)
    lg=moe.gate(x.astype(mx.float32)); inds=mx.argpartition(-lg,5,axis=-1)[...,:6]; w=mx.softmax(mx.take_along_axis(lg,inds,-1),-1)
    ref=lambda:(sw(x,inds)*w[...,None].astype(mx.bfloat16)).sum(-2)
    ik=inds.reshape(Mt,6).astype(mx.uint32)
    new=lambda:K.moe(x.reshape(Mt,2048),ik,w.reshape(Mt,6).astype(mx.float32),sw.gate_proj,sw.up_proj,sw.down_proj).reshape(1,Mt,2048)
    a=ref();b=new();mx.eval(a,b)
    err=mx.abs(a.astype(mx.float32)-b.astype(mx.float32)).max().item();sc=mx.abs(a.astype(mx.float32)).max().item()
    print(Mt,"err",err,"scale",sc,"ref ms",bench(ref),"new ms",bench(new),flush=True)
