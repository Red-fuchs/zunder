import sys,time,os,numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download
tp=snapshot_download("RecursiveSI/flint-int4")
model,tok=load(tp)
ids=tok.encode("Apple silicon unified memory lets the GPU read weights. "*30)[:256]
c=make_prompt_cache(model);mx.eval(model(mx.array([ids]),cache=c));mx.eval([x.state for x in c])
rs=np.random.RandomState(0)
def bench(f,n=15):
    for _ in range(3):f()
    t=time.perf_counter()
    for _ in range(n):f()
    return (time.perf_counter()-t)/n*1000
for M in (1,5,8):
    x=mx.array(rs.randint(1000,5000,(1,M)))
    def st():
        mx.eval(model(x,cache=c));[k.trim(M) for k in c if k.is_trimmable()]
    print("verify",M,f"{bench(st):.1f} ms",flush=True)
# component breakdown at M=5: time each part across all layers by eval-isolated calls
L=model.layers if hasattr(model,'layers') else model.model.layers
for M in (1,5):
    h=mx.random.normal((1,M,2048)).astype(mx.bfloat16)
    tot={}
    from collections import defaultdict
    tot=defaultdict(float)
    for i,l in enumerate(L):
        for nm in ('self_attn','mlp'):
            m=getattr(l,nm,None)
            if m is None: continue
            if nm=='self_attn':
                mask=None
                f=lambda m=m,i=i:mx.eval(m(h,mask="causal" if M>1 else None,cache=c[i]))
            else:
                f=lambda m=m:mx.eval(m(h))
            try:
                t=bench(f,5)
            except Exception as e:
                print("err",nm,e);continue
            if nm=='self_attn': [k.trim(0) for k in []]
            kind=nm if nm=='self_attn' else ('moe' if hasattr(m,'switch_mlp') else 'densemlp')
            tot[kind]+=t
    print("M",M,dict(tot),flush=True)
    if M>1:
        for l in L[2:3]:
            m=l.mlp
            hh=h
            inds=mx.random.randint(0,256,(1,M,6))
            print("switch_mlp",bench(lambda:mx.eval(m.switch_mlp(hh,inds)),10),"shared",bench(lambda:mx.eval(m.shared_expert(hh)),10),"gate",bench(lambda:mx.eval(m.gate(hh.astype(mx.float32))),10))
