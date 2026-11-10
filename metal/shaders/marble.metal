// ZcTaskMarble on the whole image: the luma path (Clarity) and the chroma
// path (chroma NR, dev+0x218 == 1 with look+0x109c != 0). Mirrors
// sony_marble.marble with region = whole image (and marble_slab_u16 in the
// Rust core for the luma path). All block and pixel math is integer; 64-bit
// products are truncated to 32 bits where the reference does so.
struct MarbleArgs {
    uint width,height,w4,h4,n8w,n8h;
    int m_fwd[9],m_inv[9];
    int thr_y,a_y,g20,g24,lo_l,hi_l,gate,g34,g38,g3c,g40;
    float f; int inset;
    // chroma path
    int lum_on,bnr,blk,nbw,nbh;
    uint d0; int cc,c8,c4,c0,bc,b8,b4,b0,ac,a8,l98,l90,l88,t7c,t80;
    int up_t[4];
};
constant long marble_recip25[26]={32768,32768,16384,10923,8192,6554,5461,4681,4096,3641,3277,2979,2731,2521,2341,2185,2048,1928,1820,1725,1638,1560,1489,1425,1365,1311};
inline int wrap32(long v) { return int(uint(ulong(v)&0xffffffffUL)); }
inline ushort marble_conv(device const ushort *r,device const ushort *g,device const ushort *b,uint i,device const short *dec,constant int *m,device const short *enc,int row) {
    long s0=dec[r[i]],s1=dec[g[i]],s2=dec[b[i]];
    long o=(long(m[3*row])*s0+long(m[3*row+1])*s1+long(m[3*row+2])*s2)>>13;
    return ushort(enc[clamp(o,0L,0x3fffL)]);
}
// Initial colour conversion and RGB -> YCbCr.
kernel void marble_in(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],device const short *dec [[buffer(3)]],device const short *enc [[buffer(4)]],device ushort *yo [[buffer(5)]],device ushort *cbo [[buffer(6)]],device ushort *cro [[buffer(7)]],constant MarbleArgs &a [[buffer(8)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    uint i=p.y*a.width+p.x;
    long rr=marble_conv(r,g,b,i,dec,a.m_fwd,enc,0),gg=marble_conv(r,g,b,i,dec,a.m_fwd,enc,1),bb=marble_conv(r,g,b,i,dec,a.m_fwd,enc,2);
    yo[i]=ushort(min((rr*0xb44+gg*0xdc3+bb*0x271)>>11,0xefffL)+0x1000);
    cbo[i]=ushort(clamp(wrap32(rr*-0x6ab+gg*-0x964+bb*0x1011)>>11,-0x8000,0x7fff)+0x8000);
    cro[i]=ushort(clamp(wrap32(rr*0x1acc+gg*-0x1ac0+bb*-9)>>11,-0x8000,0x7fff)+0x8000);
}
// Luma padded to w4 x h4: right edge repeats the last column; padded rows
// repeat the last real row with each even column duplicated.
inline int marble_yc(device const ushort *y,int x,int yy,constant MarbleArgs &a) {
    int rw=int(a.width),rh=int(a.height);
    if(yy>=rh){yy=rh-1;x&=~1;}
    if(x>=2*(rw/2))x=rw-1;
    return int(y[yy*rw+x]);
}
kernel void marble_blocks(device const ushort *y [[buffer(0)]],device int *m [[buffer(1)]],constant MarbleArgs &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.n8w||p.y>=a.n8h)return;
    int y0=int(p.y)*8,y1=min(y0+8,int(a.h4)),x0=int(p.x)*8,x1=min(x0+8,int(a.w4)),s=0;
    for(int yy=y0;yy<y1;yy++)for(int x=x0;x<x1;x++)s+=marble_yc(y,x,yy,a);
    m[p.y*a.n8w+p.x]=max((s+0x20)>>6,1);
}
// 5x5 (step 2) sigma filter on the edge-replicated block grid, written into
// a grid padded by 6 blocks; computed over 4 extra blocks on every side.
kernel void marble_sigma(device const int *m [[buffer(0)]],device int *q [[buffer(1)]],constant MarbleArgs &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.n8w+4||p.y>=a.n8h+4)return;
    int nw=int(a.n8w),nh=int(a.n8h),cx=int(p.x)+4,cy=int(p.y)+4;
    #define PADM(px,py) m[clamp((py)-6,0,nh-1)*nw+clamp((px)-6,0,nw-1)]
    int c=PADM(cx,cy);long acc=0;int n=0;
    for(int dy=-4;dy<=4;dy+=2)for(int dx=-4;dx<=4;dx+=2){int v=PADM(cx+dx,cy+dy);if(abs(c-v)<=a.thr_y&&v!=0){acc+=v;n++;}}
    #undef PADM
    int idx=uint(n-1)>24u?0:n;
    uint u=uint(ulong(marble_recip25[idx]*acc+0x4000)&0xffffffffUL);
    q[uint(cy)*(a.n8w+12)+uint(cx)]=u>=0x80000000u?0xffff:int((u>>15)&0xffffu);
}
kernel void marble_binom(device const int *q [[buffer(0)]],device int *o [[buffer(1)]],constant MarbleArgs &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.n8w||p.y>=a.n8h)return;
    int pw=int(a.n8w)+12,i=(6+int(p.y))*pw+6+int(p.x);
    #define Q(dy,dx) long(q[i+(dy)*pw+(dx)])
    long sum=Q(-1,-1)+2*Q(-1,0)+Q(-1,1)+2*Q(0,-1)+4*Q(0,0)+2*Q(0,1)+Q(1,-1)+2*Q(1,0)+Q(1,1);
    ulong t=ulong(sum*(0x100-long(a.a_y))+long(a.a_y)*0x10*Q(0,0)+0x800)&0xffffffffUL;
    #undef Q
    o[p.y*a.n8w+p.x]=int((t>>12)&0xffffUL);
}
inline long marble_q(device const int *q,int bx,int by,constant MarbleArgs &a) {
    return (bx<int(a.n8w)&&by<int(a.n8h))?long(q[by*int(a.n8w)+bx]):0L;   // beyond the grid reads 0
}
inline ushort marble_half(device const ushort *c,uint row,int x,int w) {
    int j=x>>1;return 2*j+1<w?ushort((int(c[row+2*j])+int(c[row+2*j+1]))>>1):c[row+w-1];
}
inline uint marble_blend(int nv,int ov,float f) {
    float o=float(ov),v=clamp(fma(float(nv)-o,f,o),0.0f,65535.0f);return uint(int(trunc(v)))&0xffffu;
}
// x8 bilinear upsample, detail gain, chroma blend, YCbCr -> RGB and the final
// colour conversion inside the 16-px valid region.
kernel void marble_out(device const ushort *y [[buffer(0)]],device const ushort *cb [[buffer(1)]],device const ushort *cr [[buffer(2)]],device const int *q [[buffer(3)]],device const ushort *r0 [[buffer(4)]],device const ushort *g0 [[buffer(5)]],device const ushort *b0 [[buffer(6)]],device const short *dec [[buffer(7)]],device const short *enc [[buffer(8)]],device ushort *ro [[buffer(9)]],device ushort *go [[buffer(10)]],device ushort *bo [[buffer(11)]],device const ushort *hcb [[buffer(12)]],device const ushort *hcr [[buffer(13)]],constant MarbleArgs &a [[buffer(14)]],uint2 p [[thread_position_in_grid]]) {
    int w=int(a.width),h=int(a.height);if(int(p.x)>=w||int(p.y)>=h)return;
    int x=int(p.x),yy=int(p.y);uint row=uint(yy*w),i=row+uint(x);
    uint hrow=uint(yy)*(a.w4/2)+uint(x>>1);
    long ylum;
    if(a.lum_on) {
    int bx=x>>3,by=yy>>3,j=x&7,k=yy&7;
    long d29=marble_q(q,bx,by,a),d30=marble_q(q,bx+1,by,a),d27=marble_q(q,bx,by+1,a),d25=marble_q(q,bx+1,by+1,a);
    long top=15-2*k,bot=1+2*k,l=15-2*j,rw=1+2*j;
    long xv=(d30*rw+d29*l)*top+bot*(d25*rw+d27*l);
    long y35=((xv+128)>>8)&0xffff;
    // detail gain
    long yo=long(y[i])>>6,d=yo-(y35>>6);
    long gain=d>=0?long(a.g20):long(a.g24),ad=abs(d);
    long a5=min(ad,long(a.g3c)),a4=ad>=long(a.g38)?a5-long(a.g38):0;
    long g2=min((((long(a.g34)*0x400+a4*long(a.g40))>>9)+1)>>1,0x400L);
    long g5=a.gate!=0?g2:0x400;
    long wlo=a.hi_l==1?yo*6:(a.hi_l==2?yo<<2:(a.hi_l==3?yo*3:0x200));
    long whi=a.lo_l==1?(yo^0x3ff)*6:(a.lo_l==2?(yo<<2)^0xffc:(a.lo_l==3?(yo^0x3ff)*3:0x200));
    long wv=long(min(ulong(wlo)&0xffffffffUL,ulong(whi)&0xffffffffUL));if(wv>0x1ff)wv=0x200;
    int v=wrap32(((((g5*gain)>>9)+1)>>1)*d*wv+yo*0x80000);
    int rr=v>0?((v>>18)+1)>>1:-(((int(0u-uint(v))>>18)+1)>>1);
    ylum=long(clamp(rr,0,0x3ff))<<6;
    } else ylum=long(y[i]);
    // chroma: the half-width planes upsampled by duplication, blended by f
    long cbv=marble_blend(int(hcb[hrow]),int(cb[i]),a.f);
    long crv=marble_blend(int(hcr[hrow]),int(cr[i]),a.f);
    long Y=ylum-0x1000,U=cbv-0x8000,V=crv-0x8000;
    ushort rgb[3];
    rgb[0]=ushort(clamp(wrap32(Y*0x2546+V*0x1526+U*-0x59e)>>15,0,0x3fff));
    rgb[1]=ushort(clamp(wrap32(Y*0x254b+V*-0x1118+U*-0x5b3)>>15,0,0x3fff));
    rgb[2]=ushort(clamp(wrap32(Y*0x2542+V*-0x136+U*0x3a10)>>15,0,0x3fff));
    if(x>=16&&x<w-16&&yy>=16&&yy<h-16) {
        long s0=dec[rgb[0]],s1=dec[rgb[1]],s2=dec[rgb[2]];
        for(int c=0;c<3;c++){long o=(long(a.m_inv[3*c])*s0+long(a.m_inv[3*c+1])*s1+long(a.m_inv[3*c+2])*s2)>>13;rgb[c]=ushort(enc[clamp(o,0L,0x3fffL)]);}
        ro[i]=rgb[0];go[i]=rgb[1];bo[i]=rgb[2];
    } else if(a.inset) { ro[i]=r0[i];go[i]=g0[i];bo[i]=b0[i]; }
    else { ro[i]=rgb[0];go[i]=rgb[1];bo[i]=rgb[2]; }
}

// Half-width chroma (FUN_10012672c): pairs averaged; past the last pair the
// last column; rows past the image repeat the last row.
kernel void marble_halfc(device const ushort *cb [[buffer(0)]],device const ushort *cr [[buffer(1)]],device ushort *hcb [[buffer(2)]],device ushort *hcr [[buffer(3)]],constant MarbleArgs &a [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    uint half4=a.w4/2;if(p.x>=half4||p.y>=a.h4)return;
    int w=int(a.width),yy=min(int(p.y),int(a.height)-1),x=int(p.x);uint row=uint(yy*w),o=p.y*half4+p.x;
    if(x<w/2){hcb[o]=ushort((int(cb[row+2*x])+int(cb[row+2*x+1]))>>1);hcr[o]=ushort((int(cr[row+2*x])+int(cr[row+2*x+1]))>>1);}
    else {hcb[o]=cb[row+w-1];hcr[o]=cr[row+w-1];}
}
// Chroma-path block means: Y over B x B, Cb / Cr over B x B/2 on the half planes.
kernel void marble_cblocks(device const ushort *y [[buffer(0)]],device const ushort *hcb [[buffer(1)]],device const ushort *hcr [[buffer(2)]],device int *my [[buffer(3)]],device int *mcb [[buffer(4)]],device int *mcr [[buffer(5)]],constant MarbleArgs &a [[buffer(6)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=uint(a.nbw)||p.y>=uint(a.nbh))return;
    int B=a.blk,y0=int(p.y)*B,y1=min(y0+B,int(a.h4)),sy=0;
    for(int yy=y0;yy<y1;yy++)for(int x=int(p.x)*B;x<min(int(p.x)*B+B,int(a.w4));x++)sy+=marble_yc(y,x,yy,a);
    int half4=int(a.w4/2),hb=B/2,sb=0,sr=0;
    for(int yy=y0;yy<y1;yy++)for(int x=int(p.x)*hb;x<min(int(p.x)*hb+hb,half4);x++){sb+=int(hcb[yy*half4+x]);sr+=int(hcr[yy*half4+x]);}
    int shy=B==8?6:4,shc=B==8?5:3;uint o=p.y*uint(a.nbw)+p.x;
    my[o]=max((sy+(1<<(shy-1)))>>shy,1);mcb[o]=max((sb+(1<<(shc-1)))>>shc,1);mcr[o]=max((sr+(1<<(shc-1)))>>shc,1);
}
inline long marble_cthr(long x,long t79,long k) {
    long pp=(x*t79)&0xffffffffL;long u=pp<0x80?1:(pp+0x80)>>8;if(pp>=0xffff80)u=0xffff;
    long v=((u*k+0x80)&0xffffffffL)>>8;if(v<0x101)v=0x100;return v>0x7fff?0x8000:v;
}
inline uint marble_sat15(long v) { uint u=uint(ulong(v)&0xffffffffUL);return u>=0x80000000u?0xffffu:(u>>15)&0xffffu; }
// Three-channel 5x5 (step 2) sigma filter of the chroma block means.
kernel void marble_csigma(device const int *my [[buffer(0)]],device const int *mcb [[buffer(1)]],device const int *mcr [[buffer(2)]],device int *qb [[buffer(3)]],device int *qr [[buffer(4)]],constant MarbleArgs &a [[buffer(5)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=uint(a.nbw+4)||p.y>=uint(a.nbh+4))return;
    int nw=a.nbw,nh=a.nbh,cx=int(p.x)+4,cy=int(p.y)+4;
    #define PAD(m,px,py) long(m[clamp((py)-6,0,nh-1)*nw+clamp((px)-6,0,nw-1)])
    long yc=PAD(my,cx,cy),ub=PAD(mcb,cx,cy),vc=PAD(mcr,cx,cy);
    long t79=clamp(long(a.c8)+long(wrap32(long(a.cc)*yc+0x80)>>8),0L,0xffffL);
    long t59=clamp(long(a.c0)+long(wrap32((ub-0x8000)*long(a.c4)+0x80)>>8),0L,0xffffL);
    long t58=clamp(long(a.b8)+long(wrap32((vc-0x8000)*long(a.bc)+0x80)>>8),0L,0xffffL);
    long tb=marble_cthr(t59,t79,a.b4),tr=marble_cthr(t58,t79,a.b0),sb=0,sr=0;int n=0;
    for(int dy=-4;dy<=4;dy+=2)for(int dx=-4;dx<=4;dx+=2){
        long yn=PAD(my,cx+dx,cy+dy),bn=PAD(mcb,cx+dx,cy+dy),rn=PAD(mcr,cx+dx,cy+dy);
        if(abs(ub-bn)<=tb&&abs(vc-rn)<=tr&&abs(yc-yn)<=long(a.d0)&&yn!=0&&bn!=0&&rn!=0){sb+=bn;sr+=rn;n++;}
    }
    #undef PAD
    int idx=uint(n-1)>24u?0:n;uint o=uint(cy)*uint(nw+12)+uint(cx);
    qb[o]=int(marble_sat15(marble_recip25[idx]*sb+0x4000));qr[o]=int(marble_sat15(marble_recip25[idx]*sr+0x4000));
}
inline long marble_cq(device const int *q,int bx,int by,constant MarbleArgs &a) {
    return (bx<a.nbw&&by<a.nbh)?long(q[by*a.nbw+bx]):0L;
}
inline long marble_up(device const int *q,device const int *upw,int bx,int by,int ii,constant MarbleArgs &a) {
    long v=long(upw[a.up_t[0]+ii])*marble_cq(q,bx,by,a)+long(upw[a.up_t[1]+ii])*marble_cq(q,bx+1,by,a)
          +long(upw[a.up_t[2]+ii])*marble_cq(q,bx+1,by+1,a)+long(upw[a.up_t[3]+ii])*marble_cq(q,bx,by+1,a);
    if(v>0x7fffbf)return 0xffff;
    return v+0x40<0?0:((v+0x40)>>7)&0xffff;
}
// FUN_100129af0: blocks upsampled to half width (offset B/2 rows, B/4 columns;
// a later block overwrites the columns it shares with the previous one), then merged.
kernel void marble_cup(device const int *qb [[buffer(0)]],device const int *qr [[buffer(1)]],device const int *upw [[buffer(2)]],device const ushort *hcr [[buffer(3)]],device ushort *ocb [[buffer(4)]],device ushort *ocr [[buffer(5)]],constant MarbleArgs &a [[buffer(6)]],uint2 p [[thread_position_in_grid]]) {
    int half4=int(a.w4/2);if(int(p.x)>=half4||p.y>=a.h4)return;
    int B=a.blk,hb=B/2,qb4=B/4,ncol=qb4+hb,c=int(p.x),r=int(p.y);
    long ub=0,vr=0;
    if(c>=qb4&&r>=hb&&(r-hb)/B<a.nbh) {
        int bx=min((c-qb4)/hb,a.nbw-1),j=c-(qb4+bx*hb);
        if(j<ncol){int by=(r-hb)/B,ii=((r-hb)%B)*hb+j;ub=marble_up(qb,upw,bx,by,ii,a);vr=marble_up(qr,upw,bx,by,ii,a);}
    }
    long av=ub<0x8000?0x8000-ub:ub-0x8000;
    long u5=max(long(wrap32((av-long(a.l98)*0x100)*long(a.t7c))),0L);if(u5>0x7fffff)u5=0x800000;
    long u29=max(long(wrap32((vr-long(a.l90)*0x100-0x8000)*long(a.t80))),0L);
    if(u29>=0x800000-u5)u29=0x800000-u5;if(u29>0x7fffff)u29=0x800000;
    long wv=((u29+0x8000)>>16)*long(a.l88);uint o=p.y*uint(half4)+p.x;
    ocr[o]=ushort(marble_sat15((0x8000-wv)*vr+wv*long(hcr[o])+0x4000));ocb[o]=ushort(ub);
}
