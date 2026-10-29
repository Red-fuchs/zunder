import sys,os,json,shutil,glob
import mlx.core as mx, mlx.nn as nn
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import model_mlx as M
from mlx.utils import tree_flatten
for bits in (4,8):
  d=M.load_draft("incoai/Flint-DFlash2")
  nn.quantize(d,group_size=64,bits=bits,class_predicate=M._dq_pred);mx.eval(d.parameters())
  out=os.path.expanduser(f"~/Projects/zunder/ref/out/draft-q{bits}");os.makedirs(out,exist_ok=True)
  src=glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--incoai--Flint-DFlash2/snapshots/*/config.json"))[0]
  cfg=json.load(open(src));cfg["quantization"]={"group_size":64,"bits":bits};json.dump(cfg,open(out+"/config.json","w"),indent=1)
  w={k:v for k,v in tree_flatten(d.parameters()) if not k.endswith('rope') and v is not None}
  # loader expects codebook keys with .weight stripped on read
  for n in ("predecessor_codebook","successor_codebook"):
    k=f"candidate_selector.{n}.weight"
    if k in w:w[f"candidate_selector.{n}"]=w.pop(k)
  mx.save_safetensors(out+"/model.safetensors",w);print(bits,len(w),sum(v.nbytes for v in w.values())/1e6,"MB")
