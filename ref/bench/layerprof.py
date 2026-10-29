import sys,os,time,collections as C,statistics as S
sys.path.insert(0,os.path.expanduser('~/Projects/zunder/ref/dflash'))
import mlx.core as mx, model_mlx as M
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download
sys.path.append(os.path.expanduser('~/.venvs/locals-bench/lib/python3.12/site-packages'))
from locals import kernels as K
model,tok=M.load(snapshot_download("RecursiveSI/flint-int4")); K.enable_small_m(os.environ.get('SMALLM','1')=='1',model)
if os.environ.get('PACK')=='1':
  sys.path.insert(0,os.path.expanduser('~/Projects/zunder/ref'));from kern import mtplx_pack;print('packed',mtplx_pack.pack(model))
lm=model.language_model if hasattr(model,'language_model') else model
layers=lm.model.layers if hasattr(lm.model,'layers') else lm.layers
acc=C.defaultdict(float)
def wrap(m,key):
  cls=type(m);orig=cls.__call__
  def f(self,*a,**k):
    t=time.perf_counter();r=orig(self,*a,**k);mx.eval(r);acc[key+':'+cls.__name__]+=time.perf_counter()-t;return r
  m.__class__=type(cls.__name__,(cls,),{'__call__':f})
for l in layers:
  for n in ('self_attn','linear_attn','mlp'):
    if hasattr(l,n):wrap(getattr(l,n),n)
ids=mx.array(tok.apply_chat_template([{"role":"user","content":"Explain hash maps."}],add_generation_prompt=True))
tc=make_prompt_cache(model);mx.eval(model(ids[None],tc))
for Mr in (1,5,8):
  acc.clear();N=15
  for i in range(N):
    mx.eval(model(mx.array([[1]*Mr]),tc))
    for c in tc:
      if c.is_trimmable():c.trim(Mr)
    if i==2:acc.clear();N=12
  print("M",Mr,{k:round(v/12*1e3,1) for k,v in acc.items()},"sum",round(sum(acc.values())/12*1e3,1),flush=True)
