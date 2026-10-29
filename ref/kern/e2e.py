import sys,os,time,numpy as np
sys.path.insert(0,os.path.expanduser('~/Projects/zunder/ref/dflash'))
import mlx.core as mx, model_mlx as Mm, kpatch
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download
model,tok=Mm.load(snapshot_download("RecursiveSI/flint-int4"))
draft=Mm.load_draft("incoai/Flint-DFlash2").bind(model)
ids=tok.encode(open(os.path.expanduser("~/Projects/zunder/ref/gen_ref.py")).read())[:300]
def fresh():
    c=make_prompt_cache(model);mx.eval(model(mx.array([ids[:256]]),cache=c));mx.eval([x.state for x in c]);return c
ref={}
for on in (False,True):
    kpatch.enable(on); c=fresh()
    for M in (1,5,8):
        x=mx.array([ids[256:256+M]]); l=model(x,cache=c); mx.eval(l)
        [k.trim(M) for k in c if k.is_trimmable()]
        lg=np.array(l.astype(mx.float32))[0]
        if not on: ref[M]=lg
        else: print("M",M,"top1 match",(lg.argmax(-1)==ref[M].argmax(-1)).all(),"max|d|",np.abs(lg-ref[M]).max(),"ref scale",np.abs(ref[M]).max(),flush=True)
        r=[]
        for _ in range(4):
            t=time.perf_counter()
            for _ in range(8): mx.eval(model(x,cache=c));[k.trim(M) for k in c if k.is_trimmable()]
            r.append((time.perf_counter()-t)/8*1000)
        print("kern" if on else "base","verify",M,f"{min(r):.1f}ms",flush=True)
P=["Write a Python function that checks if a number is prime, with docstring and tests.","Solve step by step: a train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed?","Explain how a hash map handles collisions, in about 150 words.","Write a haiku sequence about autumn rain and then translate it into German."]
for on in (False,True):
    kpatch.enable(on)
    for p in P:
        i=mx.array(tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True));acc=[]
        for r in Mm.stream_generate(model,draft,tok,i,block_size=5,max_tokens=200):
            last=r
            if r.accepted is not None: acc.append(r.accepted)
        m=sum(acc)/len(acc)
        print("kern" if on else "base",p[:20],f"tok/step={m+1:.2f} tps={last.generation_tps:.1f} ms/step={(m+1)/last.generation_tps*1000:.0f}",flush=True)
