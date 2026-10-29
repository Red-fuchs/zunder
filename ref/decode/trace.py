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
N=48
# A pipelined
c,y=prime();t=pc()
for _ in range(N): y=step(c,y);mx.async_eval(y)
mx.eval(y);A=(pc()-t)/N*1e3
# B sync per token, split build vs eval
c,y=prime();bs=[];es=[]
for _ in range(N):
  t=pc();yn=step(c,y);t1=pc();mx.eval(yn);t2=pc();bs.append(t1-t);es.append(t2-t1);y=yn
B=S.median(bs)*1e3;E=S.median(es)*1e3
# C chain 8 tokens lazily, one eval
c,y=prime();ct=[];cb=[]
for _ in range(6):
  t=pc()
  for _ in range(8): y=step(c,y)
  t1=pc();mx.eval(y);t2=pc();cb.append((t1-t)/8);ct.append((t2-t)/8)
Cb=S.median(cb)*1e3;Ct=S.median(ct)*1e3
print(f'A pipelined {A:.2f} ms/tok ({1e3/A:.1f} tok/s)')
print(f'B sync: build {B:.2f} ms, eval(wait) {E:.2f} ms, total {B+E:.2f}')
print(f'C chain8: build {Cb:.2f} ms/tok, total {Ct:.2f} ms/tok')
# GPU-bound estimate: eval wait time when build overlapped is ~GPU busy; chain total ~ max(build,gpu)+tail
# D compile
try:
  if os.environ.get('COMP')!='1': raise Exception('skipped')
  c,y=prime()
  st=[c[i].state for i in range(len(c))] if False else c
  f=mx.compile(lambda y:step(c,y),inputs=c,outputs=c,shapeless=False)
  for _ in range(3): y=f(y);mx.eval(y)
  t=pc()
  for _ in range(N): y=f(y);mx.async_eval(y)
  mx.eval(y);print(f'D compile: {(pc()-t)/N*1e3:.2f} ms/tok')
except Exception as e: print('D compile failed',repr(e)[:300])
# E trace
if os.environ.get('TRACE')=='1':
  c,y=prime()
  for _ in range(4): y=step(c,y);mx.eval(y)
  p=os.path.expanduser('~/Projects/zunder/ref/out/flint.gputrace')
  import shutil;shutil.rmtree(p,ignore_errors=True)
  mx.metal.start_capture(p)
  for _ in range(4): y=step(c,y);mx.eval(y)
  mx.metal.stop_capture();print('trace',p)
