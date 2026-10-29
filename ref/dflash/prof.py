import sys,time,os,statistics as S
sys.path.insert(0,os.path.dirname(__file__))
import mlx.core as mx, model_mlx as M
from mlx_lm.generate import generation_stream
from huggingface_hub import snapshot_download
sys.path.append(os.path.expanduser('~/.venvs/locals-bench/lib/python3.12/site-packages'))
from locals import kernels as K
tp=snapshot_download("RecursiveSI/flint-int4")
model,tok=M.load(tp); K.enable_small_m(True,model)
draft=M.load_draft("incoai/Flint-DFlash2").bind(model)
from mlx_lm.models.cache import make_prompt_cache
p="Write a Python function that checks if a number is prime, with docstring and tests."
ids=mx.array(tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True))
M._patch_model(model,draft.config.target_layer_ids)
tc=make_prompt_cache(model); dc=make_prompt_cache(draft); draft.bind(model)
cap=M._GDNStateCapture()
logits,hidden,off=M._prefill_target(model,ids,tc,None,2048)
for c in dc:c.offset=off
mx.eval(logits,hidden)
tokens=[mx.argmax(logits[:,-1:],-1)[0,0].item()];n=1;bs=int(os.environ.get("BS",5));mask=int(draft.config.mask_token_id)
T={k:[] for k in "draft verify host rollback total".split()}
def tm():return time.perf_counter()
for it in range(80):
  t0=tm()
  with mx.stream(generation_stream):
    block=mx.array([[tokens[-1]]+[mask]*(bs-1)])
    dt,di,dp=draft.propose(block,hidden,dc,0.0,logits_start=1)
    if (tn:=dc[0].offset-(ids.size+n-1))>0:M._trim_recent_cache(dc,tn)
  mx.eval(dt);t1=tm()
  cap.clear()
  with mx.stream(generation_stream):
    vi=mx.concatenate([mx.array([[tokens[-1]]]),dt],axis=1)
    lg=model(vi,tc);hidden=mx.concatenate(model._hidden_states,axis=-1);tt=mx.argmax(lg,-1)
  mx.eval(tt,hidden);t2=tm()
  d=dt[0].tolist();t=tt[0].tolist()
  a=next((i for i in range(len(d)) if d[i]!=t[i]),len(d));new=d[:a]+[t[a]]
  tokens+=new;n+=len(new);t3=tm()
  tr=bs-a-1
  if tr>0:cap.rollback(tc,a,tr)
  hidden=hidden[:,:a+1,:]
  mx.eval([c.state for c in tc],hidden);t4=tm()
  if it>=5:
    for k,v in zip(T,[t1-t0,t2-t1,t3-t2,t4-t3,t4-t0]):T[k].append(v*1000)
print({k:(round(S.median(v),1),round(min(v),1)) for k,v in T.items()},"acc",n/80)

# isolated verify bench
import mlx_lm.models.gated_delta as G
def bench(label):
  ts=[]
  for i in range(25):
    cap.clear()
    vi=mx.array([[1]*5]);t=tm()
    lg=model(vi,tc);mx.eval(lg);ts.append((tm()-t)*1000)
    for c in tc:
      if c.is_trimmable():c.trim(5)
      else:pass
  print(label,round(S.median(ts),1),round(min(ts),1))
bench("verify5 with capture")
cap.close()
bench("verify5 no capture(stock gdn)")
