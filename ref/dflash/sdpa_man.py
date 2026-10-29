import time,statistics as S,mlx.core as mx
R=40
def T(f):
  mx.eval(f());r=[]
  for _ in range(5):
    t=time.perf_counter();mx.eval(f());r.append((time.perf_counter()-t)*1000/R)
  return S.median(r)
def fold(q,k,v,sk,bm,sc):
  B,H,M,D=q.shape;Hk=k.shape[1];G=H//Hk
  qf=(q*sc).reshape(B,Hk,G*M,D)
  s=qf@k.swapaxes(-1,-2)                      # B,Hk,G*M,Lk
  s=s.reshape(B,Hk,G,M,-1)
  s=mx.where(bm,s,-mx.inf)
  m=mx.maximum(s.max(-1,keepdims=True),sk.reshape(1,Hk,G,1,1).astype(s.dtype))
  e=mx.exp(s-m);den=e.sum(-1,keepdims=True)+mx.exp(sk.reshape(1,Hk,G,1,1).astype(s.dtype)-m)
  p=(e/den).astype(q.dtype).reshape(B,Hk,G*M,-1)
  return (p@v).reshape(B,H,M,D)
for Lk in (234,1200):
  M=5
  q=mx.random.normal((1,32,M,128)).astype(mx.bfloat16);k=mx.random.normal((1,4,Lk,128)).astype(mx.bfloat16);v=mx.random.normal((1,4,Lk,128)).astype(mx.bfloat16)
  sk=mx.random.normal((32,)).astype(mx.bfloat16);bm=(mx.arange(Lk-M,Lk)[:,None]>=mx.arange(Lk)[None])
  ref=mx.fast.scaled_dot_product_attention(q,k,v,scale=128**-.5,mask=bm,sinks=sk)
  o=fold(q,k,v,sk,bm,128**-.5);print('err',mx.abs(ref-o).max().item())
  print(Lk,'fold',round(T(lambda:mx.stack([fold(q+i*1e-3,k,v,sk,bm,128**-.5) for i in range(R)])),3),
   'fast',round(T(lambda:mx.stack([mx.fast.scaled_dot_product_attention(q+i*1e-3,k,v,scale=128**-.5,mask=bm,sinks=sk) for i in range(R)])),3))
