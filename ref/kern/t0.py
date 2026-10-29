import mlx.core as mx,time
x=mx.random.normal((1,2048)).astype(mx.bfloat16);mx.eval(x)
def chain(n):
    y=x
    for _ in range(n): y=y*1.0001+0.1
    return y
for n in (10,100,400):
    r=[]
    for _ in range(5):
        t=time.perf_counter();mx.eval(chain(n));r.append((time.perf_counter()-t)/n/2*1e6)
    print(n,"us per kernel",min(r))
