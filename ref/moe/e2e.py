# usage: e2e.py mode[,mode...]  modes: base zcomb zswiglu zboth catgu catgu_z
import sys,os,time,statistics as S,mlx.core as mx,mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import flint as F
from huggingface_hub import snapshot_download
modes=sys.argv[1].split(',')
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
ids=tok.encode(open(os.path.expanduser('~/Projects/zunder/ref/flint.py')).read()*20)
Z=None
if any(m!='base' for m in modes):
  import zmlx.kernels.moe as zm, zmlx.kernels.transformer as zt
  from zmlx.kernels.transformer import _swiglu_fwd_kernel
def zswiglu(gate,up):  # silu(gate)*up via ZMLX kernel on concatenated input
  D=gate.shape[-1];x=mx.concatenate([gate,up],-1)
  return _swiglu_fwd_kernel(D)(x,template=[("T",x.dtype)],grid=(x.size//2,1,1),output_shapes=[gate.shape],output_dtypes=[x.dtype])[0]
def zswiglu_cat(gu):
  D=gu.shape[-1]//2
  return _swiglu_fwd_kernel(D)(gu,template=[("T",gu.dtype)],grid=(gu.size//2,1,1),output_shapes=[gu.shape[:-1]+(D,)],output_dtypes=[gu.dtype])[0]
orig_moe=F.SparseMoeBlock.__call__
orig_glu=F.SwitchGLU.__call__
def gq(x,w,idx): return mx.gather_qmm(x,w[0],w[1],w[2],rhs_indices=idx,transpose=True,group_size=64,bits=4)
def moe_call(comb):
  def f(self,x):
    if x.shape[1]>1: return orig_moe(self,x)
    logits=self.gate(x.astype(mx.float32))
    inds,w=F.expert_routing(logits,self._zero_bias,top_k=self.top_k,multiplier=1.0,score_eps=0.0,route_scale=1.0)
    y=self.switch_mlp(x,inds)
    y=zm.moe_combine(y,w) if comb else (y*w[...,None]).sum(-2).astype(y.dtype)
    return y+self.shared_expert(x)
  return f
def glu_call(mode):
  def f(self,x,inds):
    if inds.size>=64: return orig_glu(self,x,inds)
    x=mx.expand_dims(x,(-2,-3))
    if hasattr(self,'_gu'):
      h=zswiglu_cat(gq(x,self._gu,inds))
    else:
      g=self.gate_proj;u=self.up_proj
      h=zswiglu(gq(x,(g.weight,g.scales,g.biases),inds),gq(x,(u.weight,u.scales,u.biases),inds))
    d=self.down_proj
    return gq(h,(d.weight,d.scales,d.biases),inds).squeeze(-2)
  return f
def apply(m):
  F.SparseMoeBlock.__call__=orig_moe;F.SwitchGLU.__call__=orig_glu
  if m in('zcomb','zboth','catgu_z'): F.SparseMoeBlock.__call__=moe_call(True)
  if m in('zswiglu','zboth'): F.SwitchGLU.__call__=glu_call(m)
  if m in('catgu_z',): F.SwitchGLU.__call__=glu_call(m)
  if m=='catgu_z':
    for l in model.model.layers:
      s=getattr(l.mlp,'switch_mlp',None)
      if s is None or hasattr(s,'_gu'): continue
      g,u=s.gate_proj,s.up_proj
      s._gu=tuple(mx.concatenate([getattr(g,a),getattr(u,a)],1) for a in('weight','scales','biases'));mx.eval(s._gu)
      for a in('weight','scales','biases'): setattr(g,a,mx.zeros((1,)));setattr(u,a,mx.zeros((1,)))
def run(ctx,n,off=0):
  c=make_prompt_cache(model);p=ids[off:]
  for i in range(0,ctx,512): mx.eval(model(mx.array([p[i:min(i+512,ctx)]]),cache=c))
  y=mx.array([[p[ctx]]]);ts=[];toks=[]
  for i in range(n):
    t=time.perf_counter();o=model(y,cache=c);y=mx.argmax(o[:,-1],-1)[None];mx.eval(y);ts.append((time.perf_counter()-t)*1000);toks.append(y.item())
  return toks,S.median(ts[5:])
ref=None;res={}
for m in modes:
  apply(m)
  for ctx in (256,2048):
    r=[run(ctx,60)[1] for _ in range(3)]
    print(f'{m:8s} ctx{ctx} {S.median(r):.2f} ms/tok = {1000/S.median(r):.1f} tok/s',flush=True)
  toks=[run(256,200,off=o*700)[0] for o in range(4)]
  if ref is None: ref=toks
  print(f'{m:8s} parity(4x200 greedy) vs {modes[0]}: {[a==b for a,b in zip(toks,ref)]}',flush=True)
