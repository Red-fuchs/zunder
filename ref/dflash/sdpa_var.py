import time,statistics as S,mlx.core as mx
R=40
def T(f):
  mx.eval(f());r=[]
  for _ in range(5):
    t=time.perf_counter();mx.eval(f());r.append((time.perf_counter()-t)*1000/R)
  return S.median(r)
for Lk in (234,1200):
 for M in (1,5):
  q=mx.random.normal((1,32,M,128)).astype(mx.bfloat16);k=mx.random.normal((1,4,Lk,128)).astype(mx.bfloat16);v=mx.random.normal((1,4,Lk,128)).astype(mx.bfloat16)
  sk=mx.random.normal((32,)).astype(mx.bfloat16)
  bm=(mx.arange(Lk-M,Lk)[:,None]>=mx.arange(Lk)[None])
  am=mx.where(bm,0,-mx.inf).astype(mx.bfloat16)
  def rep(**kw):
    return lambda:mx.stack([mx.fast.scaled_dot_product_attention(q+i*1e-3,k,v,scale=128**-.5,**kw) for i in range(R)])
  vs={'causal+sink':rep(mask="causal" if M>1 else None,sinks=sk),'bool+sink':rep(mask=bm,sinks=sk),'add+sink':rep(mask=am,sinks=sk),
   'causal nosink':rep(mask="causal" if M>1 else None),'bool nosink':rep(mask=bm)}
  print(Lk,M,{n:round(T(f),3) for n,f in vs.items()},flush=True)
