import time,statistics as S,mlx.core as mx,mlx.nn as nn
from mlx_lm import load
from huggingface_hub import snapshot_download
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
layers=[l.mlp for l in model.model.layers if hasattr(l.mlp,'switch_mlp')]
print('moe layers',len(layers))
D=2048;K=6
sm=layers[0].switch_mlp; g,u,d=sm.gate_proj,sm.up_proj,sm.down_proj
print(type(g).__name__,g.weight.shape,g.scales.shape,g.group_size,g.bits)
def bench(f,n=10):
  for _ in range(3): mx.eval(f())
  ts=[]
  for _ in range(7):
    t=time.perf_counter()
    o=[f() for _ in range(n)]
    mx.eval(o); ts.append((time.perf_counter()-t)/n)
  return S.median(ts)*1000
mx.random.seed(1)
NL=len(layers)
ids=[mx.random.randint(0,256,(1,1,K)).astype(mx.uint32) for _ in range(NL)]
xs=mx.random.normal((1,1,D)).astype(mx.bfloat16)
def qm(x,m,idx):
  return mx.gather_qmm(x,m.weight,m.scales,m.biases,rhs_indices=idx,transpose=True,group_size=m.group_size,bits=m.bits)
def stock(l,x,idx):
  s=layers[l].switch_mlp; return s(x,idx)
def stock_raw(l,x,idx):
  s=layers[l].switch_mlp;x=mx.expand_dims(x,(-2,-3))
  return s.down_proj(s.activation(qm(x,s.up_proj,idx),qm(x,s.gate_proj,idx)),idx) if False else None
# prebuilt variant weights
cat=[];sp={2:[],4:[]}
for l in layers:
  s=l.switch_mlp
  cat.append(tuple(mx.concatenate([getattr(s.gate_proj,a),getattr(s.up_proj,a)],axis=1) for a in('weight','scales','biases')))
  for n in (2,4):
    sp[n].append([tuple(mx.contiguous(t) for t in (getattr(p,a)[:,i*(512//n):(i+1)*(512//n)] for a in('weight','scales','biases'))) for p in (s.gate_proj,s.up_proj) for i in range(n)])
mx.eval(cat,sp)
def gq(x,w,idx): return mx.gather_qmm(x,w[0],w[1],w[2],rhs_indices=idx,transpose=True,group_size=64,bits=4)
def act(a,b): return nn.silu(a)*b  # SwiGLU(x_up, x_gate)? checked below
def v_stock(l,x,idx): return layers[l].switch_mlp(x,idx)
def mk_rep(R):
  def f(l,x,idx):
    s=layers[l].switch_mlp
    xr=mx.broadcast_to(x,(R,)+x.shape[1:]) if False else mx.repeat(x,R,axis=0).reshape(R,1,D)
    ir=mx.repeat(idx.reshape(1,K),R,axis=0).reshape(R,1,K)
    return s(xr,ir)[0:1]
  return f
def v_cat(l,x,idx):
  s=layers[l].switch_mlp;x=mx.expand_dims(x,(-2,-3))
  gu=gq(x,cat[l],idx); gt,up=gu[...,:512],gu[...,512:]
  h=s.activation(up,gt)  # same arg order as stock
  return s.down_proj(h,idx).squeeze(-2)
def mk_split(n):
  def f(l,x,idx):
    s=layers[l].switch_mlp;x=mx.expand_dims(x,(-2,-3));w=sp[n][l]
    gt=mx.concatenate([gq(x,w[i],idx) for i in range(n)],-1)
    up=mx.concatenate([gq(x,w[n+i],idx) for i in range(n)],-1)
    return s.down_proj(s.activation(up,gt),idx).squeeze(-2)
  return f
def v_batched(l,x,idx):  # 6 batch rows each M=1, idx (6,)
  s=layers[l].switch_mlp;xb=mx.broadcast_to(x.reshape(1,1,1,D),(K,1,1,D));ib=idx.reshape(K,1)
  def g1(m,x): return mx.gather_qmm(x,m.weight,m.scales,m.biases,rhs_indices=ib,transpose=True,group_size=64,bits=4)
  h=s.activation(g1(s.up_proj,xb),g1(s.gate_proj,xb))
  return g1(s.down_proj,h).reshape(1,1,K,D)
V={'stock':v_stock,'rep2':mk_rep(2),'rep4':mk_rep(4),'cat_gu':v_cat,'splitN2':mk_split(2),'splitN4':mk_split(4),'batched6':v_batched}
ref=[v_stock(l,xs,ids[l]) for l in range(NL)]
for name,f in V.items():
  try:
    o=[f(l,xs,ids[l]) for l in range(NL)]
    ok=all(mx.allclose(a.reshape(-1),b.reshape(-1),atol=2e-2,rtol=2e-2).item() for a,b in zip(o,ref))
    eq=all(mx.array_equal(a.reshape(-1),b.reshape(-1)).item() for a,b in zip(o,ref))
    t=bench(lambda:[f(l,xs,ids[l]) for l in range(NL)])
    print(f'{name:9s} {t:.3f} ms/47 layers  ({t/NL*1000:.0f} us/layer, {NL*10.6/t:.0f} GB/s) allclose={ok} bitexact={eq}',flush=True)
  except Exception as e: print(name,'ERR',repr(e)[:200])
# raw gather_qmm only, up_proj x3 per layer
t=bench(lambda:[s for l in range(NL) for s in (qm(mx.expand_dims(xs,(-2,-3)),layers[l].switch_mlp.up_proj,ids[l]),)]);print('up-only',t)
