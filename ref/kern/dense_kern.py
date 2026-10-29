import mlx.core as mx
_S="""
const uint lane=thread_index_in_simdgroup; const uint sg=simdgroup_index_in_threadgroup;
const uint nb=threadgroup_position_in_grid.x;
const uint rb=nb*(8)+sg*4;
for(uint c0=0;c0<M;c0+=CH){
  float acc[4][CH];
  _Pragma("clang loop unroll(full)") for(uint r=0;r<4;r++) _Pragma("clang loop unroll(full)") for(uint t=0;t<CH;t++)acc[r][t]=0;
  for(uint it=0;it<K/256;it++){
    uint w[4]; float s[4],b[4];
    _Pragma("clang loop unroll(full)") for(uint r=0;r<4;r++){
      uint row=min((uint)(rb+r),(uint)(N-1)); w[r]=wq[row*(K/8)+it*32+lane]; uint si=row*(K/64)+it*4+lane/8; s[r]=sc[si];b[r]=bi[si];
    }
    float wf[4][8];
    _Pragma("clang loop unroll(full)") for(uint r=0;r<4;r++) _Pragma("clang loop unroll(full)") for(uint i=0;i<8;i++) wf[r][i]=fma(s[r],(float)((w[r]>>(4*i))&0xF),b[r]);
    _Pragma("clang loop unroll(full)") for(uint t=0;t<CH;t++){
      bool ok=c0+t<M; uint m=ok?c0+t:0;
      float xv[8];
      _Pragma("clang loop unroll(full)") for(uint i=0;i<8;i++) xv[i]=ok?(float)x[m*K+it*256+lane*8+i]:0.0f;
      _Pragma("clang loop unroll(full)") for(uint r=0;r<4;r++){
        float d=0; _Pragma("clang loop unroll(full)") for(uint i=0;i<8;i++)d=fma(xv[i],wf[r][i],d);
        acc[r][t]+=d;
      }
    }
  }
  _Pragma("clang loop unroll(full)") for(uint r=0;r<4;r++) _Pragma("clang loop unroll(full)") for(uint t=0;t<CH;t++){
    float v=simd_sum(acc[r][t]);
    if(lane==0 && c0+t<M && rb+r<N) out[(c0+t)*N+rb+r]=(bfloat)v;
  }
}
"""
_k=mx.fast.metal_kernel("dq4",["x","wq","sc","bi"],["out"],_S)
def qmv(x,ql,CH=4):
    """x [...,M,K] bf16; ql QuantizedLinear 4bit g64 (no bias)"""
    sh=x.shape; K=sh[-1]; Mt=x.size//K; N=ql.weight.shape[0]
    o=_k(inputs=[x.reshape(Mt,K),ql.weight,ql.scales,ql.biases],template=[("M",Mt),("K",K),("N",N),("CH",min(CH,Mt))],
      grid=(64*((N+7)//8),1,1),threadgroup=(64,1,1),output_shapes=[(Mt,N)],output_dtypes=[mx.bfloat16])[0]
    return o.reshape(*sh[:-1],N)
