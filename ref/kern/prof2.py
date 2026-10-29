import sys,time,os,numpy as np
sys.path.insert(0,os.path.expanduser('~/Projects/zunder/ref/dflash'))
import mlx.core as mx, model_mlx as M
from huggingface_hub import snapshot_download
tp=snapshot_download("RecursiveSI/flint-int4")
model,tok=M.load(tp); draft=M.load_draft("incoai/Flint-DFlash2").bind(model)
from mlx_lm.models.cache import make_prompt_cache
ids=tok.encode("Apple silicon unified memory lets the GPU read weights. "*30)[:256]
c=make_prompt_cache(model);mx.eval(model(mx.array([ids]),cache=c));mx.eval([x.state for x in c])
rs=np.random.RandomState(0)
for m in range(1,9):
    x=mx.array(rs.randint(1000,5000,(1,m)))
    def st():
        mx.eval(model(x,cache=c));[k.trim(m) for k in c if k.is_trimmable()]
    for _ in range(3):st()
    t=time.perf_counter()
    for _ in range(10):st()
    print("verify",m,f"{(time.perf_counter()-t)*100:.1f}ms",flush=True)
# draft time
P=ids[:20]
T={'d':0,'n':0}
orig=draft.propose
def prop(*a,**k):
    t=time.perf_counter();r=orig(*a,**k);mx.eval(r[0]);T['d']+=time.perf_counter()-t;T['n']+=1;return r
draft.propose=prop
t0=time.perf_counter()
for r in M.stream_generate(model,draft,tok,mx.array(P),block_size=5,max_tokens=100):pass
tot=time.perf_counter()-t0
print("steps",T['n'],"draft ms/step",T['d']/T['n']*1000,"total ms/step",tot/T['n']*1000)
