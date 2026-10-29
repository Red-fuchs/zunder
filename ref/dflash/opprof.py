import sys,time,os,statistics as S
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import scaled_dot_product_attention as sdpa, create_attention_mask
from mlx_lm.kernels import expert_routing
from huggingface_hub import snapshot_download
model,tok=load(snapshot_download("RecursiveSI/flint-int4"))
L=model.layers
ids=tok.encode("Apple silicon unified memory lets the GPU read weights. "*30)[:230]
c=make_prompt_cache(model);mx.eval(model(mx.array([ids]),cache=c));mx.eval([x.state[0] for x in c])
def T(f,n=10):
  mx.eval(f());mx.eval(f())
  r=[]
  for _ in range(3):
    t=time.perf_counter()
    for _ in range(n):mx.eval(f())
    r.append((time.perf_counter()-t)/n*1000)
  return S.median(r)
def manual(q,k,v,sc,mask,sinks):
  B,H,Lq,D=q.shape;Hk=k.shape[1];q=q.reshape(B,Hk,H//Hk,Lq,D);k=k[:,:,None];v=v[:,:,None]
  s=(q*sc)@k.swapaxes(-1,-2)
  if mask is not None:
    if isinstance(mask,str):
      Lk=k.shape[-2];m=mx.arange(Lk-Lq,Lk)[:,None]>=mx.arange(Lk)[None];s=mx.where(m,s,-mx.inf)
    else:s=mx.where(mask,s,-mx.inf)
  sk=sinks.reshape(1,Hk,H//Hk,1,1)*mx.ones(s.shape[:-1]+(1,),s.dtype)
  s=mx.concatenate([sk.astype(s.dtype),s],-1);p=mx.softmax(s.astype(mx.float32),-1).astype(q.dtype)[...,1:]
  return (p@v).reshape(B,H,Lq,D)
out={}
for li,kind in ((0,'full'),(2,'swa')):
  l=L[li];a=l.self_attn;ca=c[li]
  print(kind,type(ca).__name__,ca.offset,flush=True)
  for M in (1,5):
    x=mx.random.normal((1,M,2048)).astype(mx.bfloat16);mx.eval(x)
    def tr():
      if ca.is_trimmable():ca.trim(M)
    d={}
    d['noop']=T(lambda:x+0)
    xn=l.input_layernorm(x)
    d['q_proj']=T(lambda:a.q_proj(xn));d['k_proj']=T(lambda:a.k_proj(xn));d['v_proj']=T(lambda:a.v_proj(xn))
    q,k,v=a.q_proj(xn),a.k_proj(xn),a.v_proj(xn)
    def qk():
      return a.q_norm(q.reshape(1,M,a.n_heads,-1)).transpose(0,2,1,3),a.k_norm(k.reshape(1,M,a.n_kv_heads,-1)).transpose(0,2,1,3),v.reshape(1,M,a.n_kv_heads,-1).transpose(0,2,1,3)
    d['qk_norm']=T(qk);qq,kk,vv=qk();mx.eval(qq,kk,vv)
    if a.rope is not None:
      d['rope']=T(lambda:(a.rope(qq,offset=ca.offset),a.rope(kk,offset=ca.offset)))
      qq=a.rope(qq,offset=ca.offset);kk=a.rope(kk,offset=ca.offset)
    def upd():
      r=ca.update_and_fetch(kk,vv);mx.eval(r);tr();return r[0][:1,:1,:1,:1]
    d['cache_upd(+trim)']=T(upd)
    K_,V_=ca.update_and_fetch(kk,vv);mx.eval(K_,V_)
    mask=create_attention_mask(x,ca,window_size=513 if kind=='swa' else None) if M>1 else None
    d['mask_build']=T(lambda:(create_attention_mask(x,ca,window_size=513 if kind=='swa' else None) if M>1 else x+0)) if M>1 else 0
    print(kind,M,'mask',type(mask).__name__,mask if isinstance(mask,str) else (mask.shape if mask is not None else None),'K',K_.shape,flush=True)
    d['sdpa_fast']=T(lambda:sdpa(qq,K_,V_,cache=ca,scale=a.scale,mask=mask,sinks=a.softmax_offset))
    d['sdpa_manual']=T(lambda:manual(qq,K_,V_,a.scale,mask,a.softmax_offset))
    o=sdpa(qq,K_,V_,cache=ca,scale=a.scale,mask=mask,sinks=a.softmax_offset).transpose(0,2,1,3).reshape(1,M,-1);mx.eval(o)
    d['o_proj']=T(lambda:a.o_proj(o))
    tr()
    d['ATTN_TOTAL']=T(lambda:(lambda r:(tr(),r)[1])(a(xn,mask,ca)))
    tr()
    out[(kind,M)]=d
mi=next(i for i,l in enumerate(L) if hasattr(l.mlp,'switch_mlp'));m=L[mi].mlp;print('moe layer',mi)
for M in (1,5):
  x=mx.random.normal((1,M,2048)).astype(mx.bfloat16);mx.eval(x);d={}
  d['noop']=T(lambda:x+0)
  d['gate_matmul_f32']=T(lambda:m.gate(x.astype(mx.float32)))
  lg=m.gate(x.astype(mx.float32));mx.eval(lg)
  d['router_kernel']=T(lambda:expert_routing(lg,m._zero_bias,top_k=m.top_k,multiplier=1.0,score_eps=0.0,route_scale=1.0))
  inds,w=expert_routing(lg,m._zero_bias,top_k=m.top_k,multiplier=1.0,score_eps=0.0,route_scale=1.0);mx.eval(inds,w)
  s=m.switch_mlp;xe=mx.expand_dims(x,(-2,-3));mx.eval(xe)
  d['argsort(sort path, if on)']=T(lambda:__import__('mlx_lm.models.switch_layers',fromlist=['x'])._gather_sort(xe,inds))
  ds=inds.size>=64
  d['gather_qmm_up']=T(lambda:s.up_proj(xe,inds));d['gather_qmm_gate']=T(lambda:s.gate_proj(xe,inds))
  xu=s.up_proj(xe,inds);xg=s.gate_proj(xe,inds);mx.eval(xu,xg)
  d['activation']=T(lambda:s.activation(xu,xg));h=s.activation(xu,xg);mx.eval(h)
  d['gather_qmm_down']=T(lambda:s.down_proj(h,inds))
  y=s(x,inds);mx.eval(y)
  d['switch_total']=T(lambda:s(x,inds))
  d['combine']=T(lambda:(y*w[...,None]).sum(axis=-2).astype(y.dtype))
  d['shared_expert']=T(lambda:m.shared_expert(x))
  d['MOE_TOTAL']=T(lambda:m(x))
  out[('moe',M)]=d
for kind in ('full','swa','moe'):
  print('==',kind)
  for k in out[(kind,1)]:print('%-26s M1 %6.3f  M5 %6.3f  d %+6.3f'%(k,out[(kind,1)][k],out[(kind,5)][k],out[(kind,5)][k]-out[(kind,1)][k]))
