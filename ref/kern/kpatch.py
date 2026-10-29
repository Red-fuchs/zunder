import mlx.nn as nn, mlx.core as mx
import dense_kern as D
_orig=nn.QuantizedLinear.__call__
def _call(self,x):
    K=x.shape[-1]; M=x.size//K
    if 2<=M<=8 and self.bits==4 and self.group_size==64 and K%256==0 and "bias" not in self and x.dtype==mx.bfloat16:
        return D.qmv(x,self,8)
    return _orig(self,x)
def enable(on=True): nn.QuantizedLinear.__call__=_call if on else _orig
