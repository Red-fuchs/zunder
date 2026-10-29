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
def gen(ids,K=0):
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
    x=mx.array([[y]+d]);g=mx.argmax(model(x,cache=c)[0],-1).tolist()
    a=0
    while a<len(d) and d[a]==g[a]: a+=1
    if len(d)-a>0: trim(c,len(d)-a)
    new=g[:a+1];steps+=1
    if d: hits+=1;acc_tot+=a
    stop=False
    for t in new:
      out.append(t);ctx.append(t)
      if t in EOS: stop=True;break
    y=ctx[-1]
    if stop: break
  dt=time.perf_counter()-t0
  return out[:N],dict(tps=len(out)/dt,steps=steps,hit=hits/steps,acc_per_draftstep=acc_tot/max(hits,1),tok_per_step=len(out)/steps)
# ctx[-1] is the last prompt token, which was NOT prefilled (prefill stops at len-1)
code=''.join(open(os.path.expanduser('~/Projects/zunder/ref/flint.py')).readlines()[:150])
def chat(p): return tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True)
args=json.dumps({"users":[{"id":i,"name":f"user{i}","email":f"user{i}@example.com","roles":["admin","dev"],"active":i%2==0} for i in range(12)],"options":{"limit":50,"sort":"name","verbose":True}},indent=1)
W={
"1 rewrite-code":chat("Here is a Python file:\n```python\n"+code+"\n```\nRewrite the complete file with exactly these changes: add a module docstring at the top, rename the first function you find to <name>_v2 everywhere it is used, and add a '# TODO: review' comment above the last function. Output the full file in one code block, nothing else."),
"2 tool-call":chat("Call the tool bulk_update with these args, changing every 'active' to true and every email domain to example.org. Output only the tool call as JSON: {\"name\":\"bulk_update\",\"arguments\":...}\nArgs:\n"+args),
"3 german-chat":chat("Erkläre mir ausführlich, warum der Himmel blau ist und warum Sonnenuntergänge rot aussehen."),
}
gen(W["3 german-chat"][:50],0)  # warmup
rows=[]
for name,ids in W.items():
  ids=list(ids);print(name,'prompt',len(ids),flush=True)
  base,sb=gen(ids,0)
  for K in ([3,5,8] if name.startswith('1') else [5]):
    o,s=gen(ids,K);par=o==base
    fd=next((i for i,(p,q) in enumerate(zip(o,base)) if p!=q),None)
    r=f"{name} K={K} plain={sb['tps']:.1f} lookup={s['tps']:.1f} x{s['tps']/sb['tps']:.2f} tok/step={s['tok_per_step']:.2f} acc/draftstep={s['acc_per_draftstep']:.2f} hit={s['hit']:.2f} newtok={len(o)} parity={'yes' if par else 'NO@'+str(fd)}"
    print(r,flush=True);rows.append(r)
open(os.path.join(os.path.dirname(__file__),'rows.txt'),'w').write('\n'.join(rows))
