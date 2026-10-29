import mlx.core as mx,time,statistics as S
def t(f,n=30):
  for _ in range(3):mx.eval(f())
  r=[]
  for _ in range(n):
    s=time.perf_counter();mx.eval(f());r.append((time.perf_counter()-s)*1e3)
  return S.median(r)
mx.random.seed(0)
for N,K in [(6144,2048),(2048,6144),(1024,2048),(150144,2048)]:
  w=mx.random.normal((N,K))*.05;q,s,b=mx.quantize(w,64,4);mx.eval(q,s,b)
  print(f"qmm N={N} K={K}",*[f"M{M}={t(lambda:mx.quantized_matmul(mx.random.normal((M,K)).astype(mx.bfloat16) if False else x,q,s,b,transpose=True,group_size=64,bits=4)) :.3f}" for M in (1,5,8) for x in [mx.random.normal((M,K)).astype(mx.bfloat16)]])
# MoE gather_qmm: 5 tokens x 6 experts
E,N,K=256,512,2048
w=mx.random.normal((E,N,K))*.05;q,s,b=mx.quantize(w,64,4);mx.eval(q,s,b)
for M in (1,5,8):
  x=mx.random.normal((M,1,1,K)).astype(mx.bfloat16);idx=mx.random.randint(0,E,(M,6)).astype(mx.uint32)
  print(f"gather M{M}",round(t(lambda:mx.gather_qmm(x,q,s,b,rhs_indices=idx,transpose=True,group_size=64,bits=4)),3))
