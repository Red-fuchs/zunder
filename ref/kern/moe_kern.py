import mlx.core as mx
TOPK=6
CH=2
_U='_Pragma("clang loop unroll(full)") '
_HDR="""
const uint lane=thread_index_in_simdgroup; const uint sg=simdgroup_index_in_threadgroup;
const uint p=threadgroup_position_in_grid.y; const uint nb=threadgroup_position_in_grid.x;
const uint e=inds[p];
for(uint q=0;q<p;q++) if(inds[q]==e) return;
uint tk[M]; uint sl[M]; uint cnt=0;
for(uint m=0;m<M;m++) for(uint j=0;j<6;j++) if(inds[m*6+j]==e){tk[cnt]=m;sl[cnt]=j;cnt++;}
"""
_GU=_HDR+"""
for(uint c0=0;c0<cnt;c0+=CH){
float ag[4][CH]; float au[4][CH];
UNR for(uint r=0;r<4;r++) UNR for(uint t=0;t<CH;t++){ag[r][t]=0;au[r][t]=0;}
for(uint it=0;it<K/256;it++){
  uint wg[4],wu[4]; float sg_[4],bg[4],su[4],bu[4];
  UNR for(uint r=0;r<4;r++){
    uint ro=e*N+nb*8+sg*4+r;
    wg[r]=wgt[ro*(K/8)+it*32+lane]; wu[r]=wup[ro*(K/8)+it*32+lane];
    uint si=ro*(K/64)+it*4+lane/8;
    sg_[r]=sgs[si];bg[r]=sgb[si];su[r]=sus[si];bu[r]=sub[si];
  }
  UNR for(uint t=0;t<CH;t++){
    float xv[8]; float sx=0;
    bool ok=c0+t<cnt; uint m=ok?tk[c0+t]:0;
    UNR for(uint i=0;i<8;i++){float v=ok?(float)x[m*K+it*256+lane*8+i]:0.0f;xv[i]=v;sx+=v;}
    UNR for(uint r=0;r<4;r++){
      float dg=0,du=0;
      UNR for(uint i=0;i<8;i++){dg+=xv[i]*(float)((wg[r]>>(4*i))&0xF);du+=xv[i]*(float)((wu[r]>>(4*i))&0xF);}
      ag[r][t]+=sg_[r]*dg+bg[r]*sx; au[r][t]+=su[r]*du+bu[r]*sx;
    }
  }
}
UNR for(uint r=0;r<4;r++) UNR for(uint t=0;t<CH;t++){
  float g=simd_sum(ag[r][t]),u=simd_sum(au[r][t]);
  if(lane==0 && c0+t<cnt){ float s=g/(1.0f+metal::exp(-g)); out[(tk[c0+t]*6+sl[c0+t])*N+nb*8+sg*4+r]=(bfloat)(s*u);}
}
}
""".replace("UNR",_U)
_DN=_HDR+"""
for(uint c0=0;c0<cnt;c0+=CH){
float ad[4][CH];
UNR for(uint r=0;r<4;r++) UNR for(uint t=0;t<CH;t++)ad[r][t]=0;
for(uint it=0;it<K/256;it++){
  uint w[4]; float s[4],b[4];
  UNR for(uint r=0;r<4;r++){
    uint ro=e*N+nb*8+sg*4+r;
    w[r]=wdn[ro*(K/8)+it*32+lane]; uint si=ro*(K/64)+it*4+lane/8; s[r]=dsc[si];b[r]=dbi[si];
  }
  UNR for(uint t=0;t<CH;t++){
    float xv[8]; float sx=0;
    bool ok=c0+t<cnt; uint hi=ok?(tk[c0+t]*6+sl[c0+t]):0;
    UNR for(uint i=0;i<8;i++){float v=ok?(float)h[hi*K+it*256+lane*8+i]:0.0f;xv[i]=v;sx+=v;}
    UNR for(uint r=0;r<4;r++){
      float d=0; UNR for(uint i=0;i<8;i++)d+=xv[i]*(float)((w[r]>>(4*i))&0xF);
      ad[r][t]+=s[r]*d+b[r]*sx;
    }
  }
}
UNR for(uint r=0;r<4;r++) UNR for(uint t=0;t<CH;t++){
  float v=simd_sum(ad[r][t]);
  if(lane==0 && c0+t<cnt){uint hi=tk[c0+t]*6+sl[c0+t]; out[hi*N+nb*8+sg*4+r]=(bfloat)(v*rw[hi]);}
}
}
""".replace("UNR",_U)
_gu=mx.fast.metal_kernel("moe_gu",["x","inds","wgt","sgs","sgb","wup","sus","sub"],["out"],_GU)
_dn=mx.fast.metal_kernel("moe_dn",["h","inds","rw","wdn","dsc","dbi"],["out"],_DN)
def moe(x,inds,rw,gp,up,dp):
    """x [M,K] bf16, inds [M,6] uint32, rw [M,6] f32 -> [M,K] bf16 (routed experts only)"""
    Mt,K=x.shape; N=gp.weight.shape[1]; KO=dp.weight.shape[1]
    P=Mt*TOPK
    h=_gu(inputs=[x,inds,gp.weight,gp.scales,gp.biases,up.weight,up.scales,up.biases],
      template=[("M",Mt),("K",K),("N",N),("CH",CH)],grid=(64*(N//8),P,1),threadgroup=(64,1,1),
      output_shapes=[(Mt*TOPK,N)],output_dtypes=[mx.bfloat16])[0]
    y=_dn(inputs=[h,inds,rw,dp.weight,dp.scales,dp.biases],
      template=[("M",Mt),("K",N),("N",KO),("CH",CH)],grid=(64*(KO//8),P,1),threadgroup=(64,1,1),
      output_shapes=[(Mt*TOPK,KO)],output_dtypes=[mx.bfloat16])[0]
    return y.reshape(Mt,TOPK,KO).sum(axis=1)
