import sys,time,os,statistics as S,mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
which=sys.argv[1]
if which=='jang':
  from jang_tools.load_jangtq import load_jangtq_model
  model,tok=load_jangtq_model(os.path.expanduser('~/AI-Model/models/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ4'))
else:
  from mlx_lm import load; from huggingface_hub import snapshot_download
  model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
print(which,mx.__version__,flush=True)
lm=getattr(model,'language_model',model)
def fwd(x,c): return model(x,cache=c)
src=open(os.path.expanduser('~/Projects/zunder/ref/flint.py')).read()*40
ids=tok.encode(src)
de="Erkläre mir auf Deutsch in einfachen Worten, warum der Himmel blau ist und wie ein Regenbogen entsteht."
def prefill(c,x):
  t=time.perf_counter()
  for i in range(0,len(x),512): mx.eval(fwd(mx.array([x[i:i+512]]),c))
  return len(x)/(time.perf_counter()-t)
def decode(c,x,n=256):
  o=fwd(mx.array([x[-1:]]),c);y=mx.argmax(o[:,-1],-1);mx.eval(y)
  out=[];t=time.perf_counter()
  for _ in range(n):
    out.append(y);o=fwd(y[None],c);y=mx.argmax(o[:,-1],-1);mx.async_eval(y)
  mx.eval(y);dt=time.perf_counter()-t
  return n/dt,tok.decode([int(v.item()) for v in out])
def verify(c,M):
  ts=[]
  for _ in range(8):
    k=[x.offset for x in c if hasattr(x,'offset')]
    t=time.perf_counter();mx.eval(fwd(mx.array([ids[:M]]),c));ts.append((time.perf_counter()-t)*1000)
    for x in c:
      if hasattr(x,'offset') and x.offset>0 and hasattr(x,'trim') and x.is_trimmable(): x.trim(M)
  return S.median(ts[2:])
# warmup
c=make_prompt_cache(model);prefill(c,ids[:64]);decode(c,ids[:64],20)
p=tok.encode(de)
c=make_prompt_cache(model);pf=prefill(c,p);d,txt=decode(c,p)
print(f'RES {which} ctx={len(p)} prefill_tps={pf:.0f} decode_tps={d:.1f} ms/tok={1000/d:.2f}');print(repr(txt[:200]),flush=True)
for ctx in (2048,8192):
  c=make_prompt_cache(model);pf=prefill(c,ids[:ctx]);d,_=decode(c,ids[:ctx])
  print(f'RES {which} ctx={ctx} prefill_tps={pf:.0f} decode_tps={d:.1f} ms/tok={1000/d:.2f}',flush=True)
  if ctx==2048:
    for M in (1,2,4,8):
      try: print(f'RES {which} verify M={M} ms={verify(c,M):.1f}',flush=True)
      except Exception as e: print('verify fail',e);break
