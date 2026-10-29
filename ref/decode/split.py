import sys,os,time,statistics as S,mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download
model,tok=load(snapshot_download("RecursiveSI/flint-int4"));mx.eval(model.parameters())
mx.set_wired_limit(mx.device_info()['max_recommended_working_set_size'])
ids=tok.encode(open(os.path.expanduser('~/Projects/zunder/ref/flint.py')).read())[:300]
def prime():
  c=make_prompt_cache(model);o=model(mx.array([ids]),cache=c);y=mx.argmax(o[:,-1],-1)[None];mx.eval(y);return c,y
pc=time.perf_counter
def step(c,y):
  o=model(y,cache=c);return mx.argmax(o[:,-1],-1)[None]
# warm
c,y=prime()
for _ in range(16): y=step(c,y);mx.eval(y)
N=24

import mlx.nn as nn
L=model.model.layers if hasattr(model,'model') else model.layers
print(type(L[5]).__name__,[k for k in L[5].children()])
c,y=prime()
x=model.model.embed_tokens(y) if hasattr(model,'model') else None
x=mx.random.normal((1,1,2048)).astype(mx.bfloat16);mx.eval(x)
def tm(f,n=5,rep=49):
  mx.eval(f(x))
  r=[]
  for _ in range(n):
    t=pc();h=x
    for _ in range(rep): h=f(h)
    mx.eval(h);r.append((pc()-t)*1e3)
  return S.median(r)
l=L[5]
ch=dict(l.children())
mlp=l.mlp;an=l.input_layernorm
print('moe x49 (ms/tok-equiv):',tm(lambda h:h+mlp(l.post_attention_layernorm(h))))
print('norm-only x49:',tm(lambda h:l.post_attention_layernorm(h)))
sa=l.self_attn
def att(h):
  return h+sa(l.input_layernorm(h),mask=None,cache=c[5])
try: print('attn x49:',tm(att,n=3))
except Exception as e: print('attn fail',repr(e)[:200])
print('lm_head:',tm(lambda h:model.lm_head(h) if hasattr(model,'lm_head') else h,rep=1))
for nm in ['switch_mlp','shared_expert','gate']:
  if hasattr(mlp,nm):
    m=getattr(mlp,nm);print(nm,'x49:',tm(lambda h:h+(m(h)[...,:2048] if nm!='gate' else mx.zeros_like(h)+m(h).sum()) ) if nm=='gate' else None)
