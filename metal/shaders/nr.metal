// ZcTaskRawNR on the whole CFA image (RGGB at even/even), as sony_rawnr.rawnr
// with region = whole image. Integer arithmetic throughout; the final blend
// is a single-rounding float32 fma, which is what the reference computes.
// The raw is read as a view: pixel (x, y) is raw[raw_origin + y*raw_stride + x]
// (raw_stride 0: a dense width-wide image), so a crop of the full sensor raw
// is read in place.
struct NrArgs { uint width,height; int off[3],gain[3],lim[3],wm[3]; float s; uint raw_stride,raw_origin; };
inline uint nr_raw_stride(constant NrArgs &a) { return a.raw_stride?a.raw_stride:a.width; }
inline int nr_channel(uint x,uint y) { return (y&1)+(x&1); }   // 0 R, 1 G, 2 B
inline void nr_split_value(device const ushort *raw,uint w,uint h,uint x,uint y,constant NrArgs &a,thread short &hpv,thread ushort &basev) {
    if(x<2||x>=w-2||y<2||y>=h-2){hpv=0;basev=0;return;}
    uint rs=nr_raw_stride(a),i=a.raw_origin+y*rs+x;
    #define R(dy,dx) int(raw[uint(int(i)+(dy)*int(rs)+(dx))])
    int c=R(0,0),k=nr_channel(x,y);
    int d2=R(-2,-2)+R(-2,2)+R(2,-2)+R(2,2),a2=R(0,-2)+R(-2,0)+R(0,2)+R(2,0),hv;
    if(k==1){int d1=R(-1,-1)+R(-1,1)+R(1,-1)+R(1,1);hv=((24*c-(d2+2*a2))-3*d1)*36>>10;}
    else hv=(12*c-(d2+2*a2))>>4;
    #undef R
    hpv=as_type<short>(ushort(uint(hv)&0xffffu));basev=ushort(clamp(a.off[k]+c-hv,0,0x7fff));
}
// High-pass and base on the image shrunk by 2.
// band = (first row, row count): the kernels run on a row band so a caller can
// process the image as its rows become available.
kernel void nr_split(device const ushort *raw [[buffer(0)]],device short *hp [[buffer(1)]],device ushort *base [[buffer(2)]],constant NrArgs &a [[buffer(3)]],constant uint2 &band [[buffer(4)]],uint2 q [[thread_position_in_grid]]) {
    uint2 p=uint2(q.x,q.y+band.x);
    uint w=a.width,h=a.height;if(p.x>=w||p.y>=h||q.y>=band.y)return;
    uint i=p.y*w+p.x;
    short hv;ushort bv;nr_split_value(raw,w,h,p.x,p.y,a,hv,bv);hp[i]=hv;base[i]=bv;
}
constant int2 nr_gmean[25]={int2(0,0),int2(-4,0),int2(0,-4),int2(0,4),int2(4,0),int2(0,-2),int2(0,2),int2(-2,0),int2(2,0),
    int2(-2,-2),int2(2,-2),int2(-2,2),int2(2,2),int2(-3,-1),int2(-3,1),int2(-1,-3),int2(-1,-1),int2(-1,1),int2(-1,3),
    int2(1,-3),int2(1,-1),int2(1,1),int2(1,3),int2(3,-1),int2(3,1)};          // (dy, dx)
// Sigma filter of the base, high-pass gain, and the strength blend on the image shrunk by 10.
kernel void nr_filter(device const ushort *raw [[buffer(0)]],device const short *hp [[buffer(1)]],device const ushort *base [[buffer(2)]],device const int *thr [[buffer(3)]],device ushort *out [[buffer(4)]],constant NrArgs &a [[buffer(5)]],constant uint2 &band [[buffer(6)]],uint2 q [[thread_position_in_grid]]) {
    uint2 p=uint2(q.x,q.y+band.x);
    uint w=a.width,h=a.height;if(p.x>=w||p.y>=h||q.y>=band.y)return;
    uint i=p.y*w+p.x,ri=a.raw_origin+p.y*nr_raw_stride(a)+p.x;
    if(p.x<10||p.x>=w-10||p.y<10||p.y>=h-10){out[i]=raw[ri];return;}
    #define B(dy,dx) int(base[uint(int(i)+(dy)*int(w)+(dx))])
    int k=nr_channel(p.x,p.y),b0=B(0,0),t=thr[k*0x8000+b0],t2=t*t,m=0;
    if(k==1){for(int j=0;j<25;j++)m+=B(nr_gmean[j].x,nr_gmean[j].y);m=m*0x29>>10;}
    else {for(int dy=-4;dy<=4;dy+=4)for(int dx=-4;dx<=4;dx+=4)m+=B(dy,dx);m=m*0x72>>10;}
    int cp=(a.wm[k]*b0+(0x400-a.wm[k])*m)>>10,acc=b0,n=1;
    for(int dy=-8;dy<=8;dy+=4)for(int dx=-8;dx<=8;dx+=4){
        if(dy==0&&dx==0)continue;
        int v=B(dy,dx),d=cp-v;if(d*d<=t2){acc+=v;n++;}
    }
    #undef B
    long tv=(long(a.gain[k])*long(hp[i]))>>8;
    tv=tv<=-long(a.lim[k])?-long(a.lim[k]):tv;tv=tv>long(a.lim[k])?long(a.lim[k]):tv;
    int o=int(clamp(tv-long(a.off[k])+long(acc/n),0L,0x3fffL)),r=int(raw[ri]);
    out[i]=ushort(int(trunc(fma(float(o-r),a.s,float(r)))));
}

