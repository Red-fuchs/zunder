import time,statistics as S,mlx.core as mx
D,I,E,K=2048,512,256,6
mx.random.seed(0)
def q(shape):
  w=mx.random.normal(shape).astype(mx.bfloat16)
  return mx.quantize(w,group_size=64,bits=4)
def bench(f,n=200):
  for _ in range(10): mx.eval(f())
  ts=[]
  for _ in range(5):
    t=time.perf_counter()
    outs=[f() for _ in range(n)]
    mx.eval(outs); ts.append((time.perf_counter()-t)/n)
  return S.median(ts)
# many distinct layers so cache doesn't help: 16 layers of weights
L=16
Wg=[q((E,I,D)) for _ in range(L)]; Wd=[q((E,D,I)) for _ in range(L)]
mx.eval(Wg,Wd)
def bytes_per_exp(W): return sum(a[0].nbytes for a in W)
bg=bytes_per_exp(Wg[0])
print('bytes/expert gate',bg)
for M in (1,2,4,8):
  idx=[mx.stack([mx.random.permutation(E)[:K] for _ in range(1)]).reshape(1,K) for _ in range(L)] if M==1 else None
  # M tokens, each K experts, distinct sets -> up to M*K unique
  ids=[mx.random.randint(0,E,(M,K)).astype(mx.uint32) for _ in range(L)]
  xs=mx.random.normal((M,1,1,D)).astype(mx.bfloat16)
  uniq=[len(set(i.flatten().tolist())) for i in ids]; u=sum(uniq)/L
  def f():
    return [mx.gather_qmm(xs,*Wg[l],rhs_indices=ids[l],transpose=True,group_size=64,bits=4) for l in range(L)]
  t=bench(f,20)/L
  print(f'gather_qmm M={M} uniq={u:.1f} {t*1e6:.1f}us  {u*bg/t/1e9:.1f} GB/s')
# dense qmm same bytes: K*3 experts worth as one matrix (K*I x D)
w,s,b=q((K*I,D)); x=mx.random.normal((1,D)).astype(mx.bfloat16); mx.eval(w,s,b)
Ws=[q((K*I,D)) for _ in range(L)]; mx.eval(Ws)
t=bench(lambda:[mx.quantized_matmul(x,*Ws[l],transpose=True,group_size=64,bits=4) for l in range(L)],20)/L
print(f'dense qmm M=1 {K*I}x{D}: {t*1e6:.1f}us {K*bg/t/1e9:.1f} GB/s')
