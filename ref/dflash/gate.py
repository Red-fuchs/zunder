import sys,time,os
sys.path.insert(0,os.path.dirname(__file__))
import mlx.core as mx, model_mlx as M
from huggingface_hub import snapshot_download
tp=snapshot_download("RecursiveSI/flint-int4"); dp=snapshot_download("incoai/Flint-DFlash2")
if os.environ.get('KERN'):
  sys.path.append(os.path.expanduser('~/.venvs/locals-bench/lib/python3.12/site-packages'))
  from locals import kernels as K
model,tok=M.load(tp)
if os.environ.get('KERN'): K.enable_small_m(True,model)
draft=M.load_draft("incoai/Flint-DFlash2").bind(model)
P=["Write a Python function that checks if a number is prime, with docstring and tests.",
"Solve step by step: a train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed?",
"Explain how a hash map handles collisions, in about 150 words.",
"Write a haiku sequence about autumn rain and then translate it into German."]
for bs in [int(a) for a in sys.argv[1:]] or [8]:
  for p in P:
    ids=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True)
    ids=mx.array(ids)
    n=0;t0=time.time();last=None
    acc=[]
    for r in M.stream_generate(model,draft,tok,ids,block_size=bs,max_tokens=200):
        last=r
        if r.accepted is not None: acc.append(r.accepted)
    m=sum(acc)/max(len(acc),1)
    print(f"bs={bs} {p[:24]!r} steps={len(acc)} mean_acc={m:.2f} tok/step={m+1:.2f} tps={last.generation_tps:.1f}",flush=True)
