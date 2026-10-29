import sys,time,os,numpy as np
import mlx.core as mx
from mlx_lm import load
import mlx_lm.models.flint as F
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download
import moe_kern as K
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
txt=open(os.path.expanduser("~/Projects/zunder/ref/gen_ref.py")).read()
ids=tok.encode(txt)[:300]
c=make_prompt_cache(model);mx.eval(model(mx.array([ids[:256]]),cache=c));mx.eval([x.state for x in c])
cont=ids[256:300]
SB=F.SparseMoeBlock; origcall=SB.__call__; AT=F.Attention; atcall=AT.__call__
def route(s,x):
    logits=s.gate(x.astype(mx.float32))
    return F.expert_routing(logits,s._zero_bias,top_k=6,multiplier=1.0,score_eps=0.0,route_scale=1.0)
def noexp(s,x):
    route(s,x); return s.shared_expert(x)
def kern(s,x):
    inds,w=route(s,x); B,L,D=x.shape
    y=K.moe(x.reshape(-1,D),inds.reshape(-1,6).astype(mx.uint32),w.reshape(-1,6).astype(mx.float32),s.switch_mlp.gate_proj,s.switch_mlp.up_proj,s.switch_mlp.down_proj).reshape(B,L,D)
    return y.astype(x.dtype)+s.shared_expert(x)
def noattn(s,x,mask=None,cache=None): return x
V={"base":(origcall,atcall),"noexp":(noexp,atcall),"noattn":(origcall,noattn),"kern":(kern,atcall),"kern_noattn":(kern,noattn)}
for name,(mc,ac) in V.items():
    SB.__call__=mc; AT.__call__=ac
    out=[]
    for M in (1,5,8):
        x=mx.array([cont[:M]])
        def st():
            mx.eval(model(x,cache=c));[k.trim(M) for k in c if k.is_trimmable()]
        for _ in range(3):st()
        r=[]
        for _ in range(4):
            t=time.perf_counter()
            for _ in range(8):st()
            r.append((time.perf_counter()-t)/8*1000)
        out.append(f"M{M}={min(r):.1f}")
    print(name,*out,flush=True)
