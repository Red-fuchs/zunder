import time,statistics as S,os,mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
txt=open(os.path.expanduser('~/Projects/zunder/ref/dflash/gate_fast.py')).read()
ids=tok.encode(txt)[:400]
c=make_prompt_cache(model);mx.eval(model(mx.array([ids[:300]]),cache=c));mx.eval([x.state[0] for x in c])
def bench(label,M=5,n=40):
  x=mx.array([ids[300:300+M]]);ts=[]
  for i in range(n):
    t=time.perf_counter();mx.eval(model(x,cache=c));ts.append((time.perf_counter()-t)*1000)
    for k in c:k.trim(M)
  print(label,'M',M,'median %.2f min %.2f'%(S.median(ts[5:]),min(ts)),flush=True)
from mlx_lm.models import flint as F
orig=F.SparseMoeBlock.__call__;cache={}
def cmp(self,x):
  f=cache.get(id(self))
  if f is None:f=cache[id(self)]=mx.compile(lambda x:orig(self,x))
  return f(x)
for r in range(3):
  F.SparseMoeBlock.__call__=orig;bench('base',5,60)
  F.SparseMoeBlock.__call__=cmp;bench('moe-compiled',5,60)
F.SparseMoeBlock.__call__=orig;bench('base',1,60);F.SparseMoeBlock.__call__=cmp;bench('moe-compiled',1,60)
