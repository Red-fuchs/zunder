import sys,time,os,json,mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache,RotatingKVCache
from huggingface_hub import snapshot_download
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
EOS={tok.eos_token_id}|set(getattr(tok,'eos_token_ids',[]) or [])
N=int(os.environ.get('NEW',350))
def trim(cache,n):
  for c in cache:
    if isinstance(c,RotatingKVCache):
      c.keys=c._temporal_order(c.keys);c.values=c._temporal_order(c.values)
      c.keys=c.keys[...,:-n,:];c.values=c.values[...,:-n,:];c.offset-=n;c._idx=c.keys.shape[2]
    else: c.trim(n)
def prefill(ids):
  c=make_prompt_cache(model)
  for i in range(0,len(ids)-1,512): mx.eval(model(mx.array([ids[i:min(i+512,len(ids)-1)]]),cache=c))
  return c
def gen(ids,K=0,TK=1):
  c=prefill(ids);ctx=list(ids);out=[];steps=hits=acc_tot=0
  y=ids[-1];t0=time.perf_counter()
  while len(out)<N:
    d=[]
    if K:
      for n in (4,3,2):
        if len(ctx)<=n: continue
        key=ctx[-n:]
        for j in range(len(ctx)-n-1,-1,-1):
          if ctx[j:j+n]==key:
            d=ctx[j+n:j+n+K];break
        if d: break
    # ctx[-1] is y; feed y + draft
    x=mx.array([[y]+d]);lg=model(x,cache=c)[0];g=mx.argmax(lg,-1).tolist()
    tk=mx.argpartition(-lg,TK-1,axis=-1)[:,:TK].tolist() if TK>1 and d else None
    a=0
    while a<len(d) and (d[a]==g[a] or (tk and d[a] in tk[a])): a+=1
    if len(d)-a>0: trim(c,len(d)-a)
    new=d[:a]+[g[a]];steps+=1
    if d: hits+=1;acc_tot+=a
    stop=False
    for t in new:
      out.append(t);ctx.append(t)
      if t in EOS: stop=True;break
    y=ctx[-1]
    if stop: break
  dt=time.perf_counter()-t0
  return out[:N],dict(tps=len(out)/dt,steps=steps,hit=hits/steps,acc_per_draftstep=acc_tot/max(hits,1),tok_per_step=len(out)/steps)

import ast
code=''.join(open(os.path.expanduser('~/Projects/zunder/ref/flint.py')).readlines()[:150])
PRE=os.environ.get('PRE','<think>\n\n</think>\n\n')
def chat(p,off=True):
  ids=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True)
  return list(ids)+(tok.encode(PRE,add_special_tokens=False) if off else [])
args=json.dumps({"users":[{"id":i,"name":f"user{i}","email":f"user{i}@example.com","roles":["admin","dev"],"active":i%2==0} for i in range(12)],"options":{"limit":50,"sort":"name","verbose":True}},indent=1)
P={
"1 code":"Here is a Python file:\n```python\n"+code+"\n```\nRewrite the complete file with exactly these changes: add a module docstring at the top, rename the first function you find to <name>_v2 everywhere it is used, and add a '# TODO: review' comment above the last function. Output the full file in one code block, nothing else.",
"2 tool":"Call the tool bulk_update with these args, changing every 'active' to true and every email domain to example.org. Output only the tool call as JSON: {\"name\":\"bulk_update\",\"arguments\":...}\nArgs:\n"+args,
"3 german":"Erkläre mir ausführlich, warum der Himmel blau ist und warum Sonnenuntergänge rot aussehen."}
def chk(name,o):
  t=tok.decode(o)
  try:
    if name[0]=='1':
      import re;m=re.search(r"```python\n(.*?)(```|$)",t,re.S);ast.parse(m.group(1));return 'ast-ok'
    if name[0]=='2':
      m=t[t.index('{'):t.rindex('}')+1];j=json.loads(m);a=j.get('arguments',j);a=json.loads(a) if isinstance(a,str) else a
      exp=json.loads(args)
      for u in exp['users']: u['active']=True;u['email']=u['email'].split('@')[0]+'@example.org'
      return 'json-ok,args=='+str(a==exp)
  except Exception as e: return 'FAIL:'+type(e).__name__
  return '-'
gen(chat(P["3 german"])[:50],0)
for name,p in P.items():
  th=gen(list(chat(p,False)),0)[0] if os.environ.get('THINKCHK','1')=='1' else []
  print(name,'THINKING-DEFAULT first300:',repr(tok.decode(th[:120])[:300]),flush=True)
  ids=chat(p);print(name,'prompt',len(ids),flush=True)
  base,sb=gen(ids,0);print(name,'plain first300:',repr(tok.decode(base)[:300]),'|',chk(name,base),flush=True)
  for K,TK in [(5,1),(5,2),(5,4)]:
    o,s=gen(ids,K,TK);m=sum(a==b for a,b in zip(o,base))/len(base)
    print(f"{name} K={K} topk={TK} plain={sb['tps']:.1f} lookup={s['tps']:.1f} x{s['tps']/sb['tps']:.2f} tok/step={s['tok_per_step']:.2f} hit={s['hit']:.2f} match={m*100:.0f}% {chk(name,o)}",flush=True)
