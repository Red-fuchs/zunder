import sys,time,os,statistics as S
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx, model_mlx as M
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
from huggingface_hub import snapshot_download
sys.path.append(os.path.expanduser('~/.venvs/locals-bench/lib/python3.12/site-packages'))
from locals import kernels as K
tp=snapshot_download("RecursiveSI/flint-int4")
model,tok=M.load(tp); K.enable_small_m(os.environ.get('SMALLM','1')=='1',model)
if os.environ.get('PACK')=='1':
  sys.path.insert(0,os.path.expanduser('~/Projects/zunder/ref'));from kern import mtplx_pack;print('packed',mtplx_pack.pack(model))
draft=M.load_draft("incoai/Flint-DFlash2").bind(model)
P=["Write a Python function that checks if a number is prime, with docstring and tests.",
"Solve step by step: a train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed?",
"Explain how a hash map handles collisions, in about 150 words.",
"Write a haiku sequence about autumn rain and then translate it into German."]
MODE=os.environ.get('MODE','fast'); BS=int(os.environ.get('BS',5)); N=200
eos=set(tok.eos_token_ids) if hasattr(tok,'eos_token_ids') else set()
M._patch_model(model,draft.config.target_layer_ids)
mask_id=int(draft.config.mask_token_id)

def plain(ids):
  tc=make_prompt_cache(model)
  with mx.stream(generation_stream):
    lg=model(ids[None],tc);t=mx.argmax(lg[:,-1],-1)
  out=[];
  for _ in range(N):
    tk=t.item();out.append(tk)
    if tk in eos:break
    with mx.stream(generation_stream):
      t=mx.argmax(model(t[None],tc)[:,-1],-1)
  return out

def spec(ids,prof=None):
  tc=make_prompt_cache(model); dc=make_prompt_cache(draft)
  assert can_trim_prompt_cache(tc)
  with mx.stream(generation_stream):
    lg,hid,off=M._prefill_target(model,ids,tc,None,2048)
    for c in dc:c.offset=off
  mx.eval(lg,hid)
  toks=[mx.argmax(lg[:,-1],-1).item()];n=1;steps=0;t0=time.perf_counter()
  plen=ids.size
  while n<N and toks[-1] not in eos:
    bs=min(BS,N-n+1)
    if bs<=1:break
    with mx.stream(generation_stream):
      blk=mx.array([[toks[-1]]+[mask_id]*(bs-1)])
      dt,_,_=draft.propose(blk,hid,dc,0.0,logits_start=1)
      if (tn:=dc[0].offset-(plen+n-1))>0:M._trim_recent_cache(dc,tn)
      if MODE=='fast':
        vi=mx.concatenate([blk[:,:1],dt],axis=1)
        lg=model(vi,tc);hid=mx.concatenate(model._hidden_states,axis=-1)
        tt=mx.argmax(lg,-1)
        # accepted count on GPU -> single small sync
        acc=mx.sum(mx.cumprod((dt==tt[:,:-1]).astype(mx.int32),axis=1))
        mx.async_eval(dt,tt,hid,acc)
      else:
        mx.async_eval(dt)
        vi=mx.concatenate([blk[:,:1],dt],axis=1)
        lg=model(vi,tc);hid=mx.concatenate(model._hidden_states,axis=-1);tt=mx.argmax(lg,-1)
        mx.async_eval(tt,hid)
    d=dt[0].tolist();t=tt[0].tolist()
    a=next((i for i in range(len(d)) if d[i]!=t[i]),len(d))
    new=(d[:a]+[t[a]])[:N-n]
    toks+=new;n+=len(new);steps+=1
    tr=bs-a-1
    if tr>0:M._trim_recent_cache(tc,tr)
    hid=hid[:,:a+1]
  dtm=time.perf_counter()-t0
  return toks,steps,dtm
res=[]
for rep in range(int(os.environ.get('REP',1))):
 for p in P:
  ids=mx.array(tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True))
  toks,steps,dtm=spec(ids)
  res.append((p[:20],steps,len(toks)/dtm,dtm/steps*1000))
if os.environ.get('CHECK'):
  for p in P:
    ids=mx.array(tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True))
    ref=plain(ids);toks,_,_=spec(ids)
    m=min(len(ref),len(toks));fd=next((i for i in range(m) if ref[i]!=toks[i]),-1);print("MATCH" if fd<0 else "DIFF",p[:20],"first_diff",fd,"/",m,flush=True)
for p in dict.fromkeys(r[0] for r in res):
  rs=[r for r in res if r[0]==p]
  print(MODE,"bs",BS,p,"steps",rs[0][1],"tps %.1f ms/step %.1f"%(S.median(r[2] for r in rs),S.median(r[3] for r in rs)))
