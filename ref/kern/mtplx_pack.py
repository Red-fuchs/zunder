"""Gate+up projection packing for MoE blocks (routed SwitchGLU + shared expert).
Idea from MTPLX (youssofal/MTPLX, Apache-2.0, mtplx/moe_packed_projections.py):
concat gate/up along the output axis once at load -> one matmul, split after.
Re-implemented here (no code copied); affine groups run along K so no requant.
"""
import mlx.core as mx

def _cat(a,b,ax):return mx.concatenate([a,b],axis=ax)

def pack(model):
  n=0
  for _,m in model.named_modules():
    if not (hasattr(m,'switch_mlp') and hasattr(m,'shared_expert')):continue
    s=m.switch_mlp;g,u,d=s.gate_proj,s.up_proj,s.down_proj
    if g.get('biases') is None or 'bias' in g:continue
    s._gu=(_cat(g.weight,u.weight,1),_cat(g.scales,u.scales,1),_cat(g.biases,u.biases,1),g.group_size,g.bits,g.weight.shape[1])
    e=m.shared_expert;eg,eu=e.gate_proj,e.up_proj
    e._gu=(_cat(eg.weight,eu.weight,0),_cat(eg.scales,eu.scales,0),_cat(eg.biases,eu.biases,0),eg.group_size,eg.bits,eg.weight.shape[0])
    mx.eval(s._gu,e._gu)
    s.__class__=type('PackedSwitchGLU',(type(s),),{'__call__':_switch})
    e.__class__=type('PackedMLP',(type(e),),{'__call__':_shared})
    n+=1
  return n

def _switch(self,x,indices):
  w,sc,b,gs,bits,h=self._gu
  x=mx.expand_dims(x,(-2,-3));idx=mx.stop_gradient(indices)
  if idx.size>=64:  # keep stock sorted path for prefill
    from mlx_lm.models.switch_layers import _gather_sort,_scatter_unsort
    x,idx,inv=_gather_sort(x,idx);srt=True
  else:inv=None;srt=False
  gu=mx.gather_qmm(x,w,sc,b,rhs_indices=idx,transpose=True,group_size=gs,bits=bits,sorted_indices=srt)
  y=self.down_proj(self.activation(gu[...,h:],gu[...,:h]),idx,sorted_indices=srt)
  if inv is not None:y=_scatter_unsort(y,inv,indices.shape)
  return y.squeeze(-2)

def _shared(self,x):
  w,sc,b,gs,bits,h=self._gu
  gu=mx.quantized_matmul(x,w,sc,b,transpose=True,group_size=gs,bits=bits)
  g=gu[...,:h];return self.down_proj(g*mx.sigmoid(g)*gu[...,h:])
