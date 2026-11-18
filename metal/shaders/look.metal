// Horizontal five-tap blur, hue index, LinearMatrix16 and MainGamma, followed
// optionally by RGB2YCC, AreaComp and ChromaSuppress.
struct LookArgs { uint width,height; float thresholds[3]; };
// The first five fields match LookArgs. k8..k11 are the shear terms already
// scaled by 1/256; w9..w11 are the low 32 bits of the ChromaSuppress constants.
struct FrontArgs {
    uint width,height; float thresholds[3];
    int c0,c1,c2; float k8,k9,k10,k11,gcr_pos,gcr_neg,gcb_pos,gcb_neg;
    float area[16]; int area_flag; int w9,w12,w8,w11;
};
// The five-tap row blur of one channel: i is that channel's element of pixel
// x in an interleaved row (device memory, or geo_front's threadgroup tile).
template<typename P>
inline float look_blur(P rgb,uint i,uint x,uint w) {
    float a,b;
    if(x==0)return fma(float(rgb[i+3]),8.0f,fma(float(rgb[i]),23.0f,float(rgb[i+6])))*0.03125f;
    if(x==1)return fma(float(rgb[i+3]),8.0f,fma(float(rgb[i]),14.0f,fma(float(rgb[i-3]),9.0f,float(rgb[i+6]))))*0.03125f;
    if(x==w-2){a=float(rgb[i+3])+float(rgb[i-3]);b=fma(float(rgb[i]),14.0f,float(rgb[i-6]))+float(rgb[i+3]);return fma(a,8.0f,b)*0.03125f;}
    if(x==w-1)return fma(float(rgb[i]),23.0f,fma(float(rgb[i-3]),8.0f,float(rgb[i-6])))*0.03125f;
    a=float(rgb[i+3])+float(rgb[i-3]);b=fma(float(rgb[i]),14.0f,float(rgb[i-6]))+float(rgb[i+6]);return fma(a,8.0f,b)*0.03125f;
}
// (ni*1024 + di) / (2*di) for 0 <= ni <= di (num is the smaller of the two
// differences, or 0), clamped to [0, 511], without 64-bit division: a float
// estimate (off by at most 1), then the exact remainder ni*1024 + di - 2*q*di.
// For di < 2^29 its true value stays within +-2^31, so 32-bit wrapping
// arithmetic gives it exactly; the loops move q onto 0 <= remainder < 2*di.
// Larger di (differences of 32768 or more) keep the 64-bit division.
inline int hue_index(int ni,int di) {
    if(di>=(1<<29))return clamp(int((long(ni)*1024L+di)/(long(di)*2L)),0,511);
    int q=int(fma(float(ni)/float(di),512.0f,0.5f));
    uint n1024=uint(ni)*1024u,d=uint(di);
    int r=int(n1024+d-uint(2*q)*d);
    while(r<0){q--;r+=2*di;}
    while(r>=2*di){q++;r-=2*di;}
    return clamp(q,0,511);
}
// LinearMatrix16 + MainGamma for the pixel whose R element is rgb[i]; x and
// width are its column and the image width (the blur's edge rules).
template<typename P>
inline int3 look_encode_at(P rgb,uint i,uint x,uint width,device const float *table,device const short *gamma,device const float *atan_tab,constant float *thresholds) {
    float r=rgb[i],g=rgb[i+1],b=rgb[i+2];
    float br=look_blur(rgb,i,x,width),bg=look_blur(rgb,i+1,x,width),bb=look_blur(rgb,i+2,x,width);
    // The blur is an exact multiple of 1/32 for uint16 input. Summing its
    // integer numerator preserves the reference's double-precision luma.
    int luma_num=int(br*32.0f)*109+int(bg*32.0f)*366+int(bb*32.0f)*37;
    float lum=float(luma_num)*0.00006103515625f;
    float aa=br-lum,bc=bb-lum,fa=abs(aa),fb=abs(bc);bool big=fa>fb,ap=!(aa<0),bp=!(bc<0);
    float offh=big?(ap?(bp?-256.0f:256.0f):(bp?768.0f:-768.0f)):(ap?(bp?0.0f:-512.0f):(bp?-1024.0f:512.0f));
    float num=big?fb:fa,den=big?(aa==0?1.0f:fa):(bc==0?1.0f:fb);
    // Differences are multiples of 1/16384. Integer division reproduces the
    // reference's float64 ratio and round-to-nearest at hue-bin boundaries.
    int index=hue_index(int(num*16384.0f),int(den*16384.0f));
    float hue=abs(atan_tab[index]+offh);int hi=int(hue);if(hi<0||hi>1022)hi=1023;
    device const float *t=table+hi*18;float sq0=r*r,sq1=g*g,sq2=b*b;int enc[3];
    for(int row=0;row<3;row++) {
        float lin=fma(t[3*row+2],b,fma(t[3*row+1],g,t[3*row]*r));
        float q=fma(t[9+3*row+2],sq2,fma(t[9+3*row+1],sq1,t[9+3*row]*sq0))*0.00006103515625f;
        float thr=thresholds[row],sum=q+lin;
        float v=q<0?lin:(q>=thr?sum-thr:(q<=-thr?sum+thr:lin));v=clamp(v,0.0f,16383.0f);
        uint gi=uint(v);float frac=v-floor(v),t0=gamma[32768+gi],t1=gamma[32768+gi+1];
        enc[row]=int(short(int(trunc(fma(t1-t0,frac,t0)))));
    }
    return int3(enc[0],enc[1],enc[2]);
}
inline int3 look_encode(device const ushort *rgb,device const float *table,device const short *gamma,device const float *atan_tab,uint2 p,uint width,constant float *thresholds) {
    return look_encode_at(rgb,(p.y*width+p.x)*3,p.x,width,table,gamma,atan_tab,thresholds);
}
kernel void look_lm_gamma(device const ushort *rgb [[buffer(0)]],device const float *table [[buffer(1)]],device const short *gamma [[buffer(2)]],device const float *atan_tab [[buffer(3)]],device short *out [[buffer(4)]],constant LookArgs &a [[buffer(5)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    int3 e=look_encode(rgb,table,gamma,atan_tab,p,a.width,a.thresholds);uint i=(p.y*a.width+p.x)*3;
    out[i]=short(e.x);out[i+1]=short(e.y);out[i+2]=short(e.z);
}
inline float area_edge(float a,float b,constant FrontArgs &f,int c0,int c1,int k0,int k1) {
    float x0=f.area[c0]+a,x1=f.area[c1]+b,v=fma(x1,f.area[k1],x0*f.area[k0]);return v<=0?0.0f:v;
}
// trunc(float64(n) + float64(x)) & 0xffff for integer n >= 0 and float x,
// without double support. x - floor(x) and x - ceil(x) are exact in float32,
// so the only float64 rounding left is a tiny negative fraction collapsing
// onto the integer above it; that happens within half a float64 ulp.
inline ushort area_out(int n,float x) {
    if(x>=0)return ushort(uint(long(n)+long(floor(x)))&0xffffu);
    float c=ceil(x),f=x-c;long k=long(n)+long(c);
    if(f==0||k<=0)return ushort(uint(k)&0xffffu);
    // k is in [1, 65535]; the float64 spacing just below k is 2^(e-52),
    // or 2^(e-53) when k is a power of two. Ties round to the even k.
    uint ku=uint(k);int e=31-int(clz(ku));float tie=ldexp(1.0f,(ku&(ku-1))==0?e-54:e-53);
    return ushort(uint(-f<=tie?k:k-1)&0xffffu);
}
inline ushort scale_chroma(ushort v,int f) {
    int t=int(uint(int(v)-0x8000)*uint(f));if(t<0)t=int(uint(t)+0xffu);
    return ushort(((uint(t)>>8)^0x8000u)&0xffffu);
}
inline ushort pack_chroma(float v) { return ushort(clamp(int(trunc(v+32768.0f)),0,0xffff)); }
inline void ycc_front(int3 enc,constant FrontArgs &a,device ushort *yo,device ushort *cro,device ushort *cbo,uint o) {
    // Only the low 32 bits of the 64-bit sum are used: wrapping 32-bit products give them.
    uint sum=uint(enc.x)*uint(a.c0)+uint(enc.y)*uint(a.c1)+uint(enc.z)*uint(a.c2);
    ushort yv=ushort((sum>>13)&0xffffu);
    int rg=enc.x-enc.y,bgc=enc.z-enc.y;float s3=float(rg),s4=float(bgc);
    float cb0=fma(rg<0?a.k9:a.k8,s3,s4),cr0=fma(bgc<0?a.k11:a.k10,s4,s3);
    float cr=(cr0*0.5f)*(cr0<0?a.gcr_neg:a.gcr_pos),cb=(cb0*0.5f)*(cb0<0?a.gcb_neg:a.gcb_pos);
    cr=max(min(cr,8191.0f),-8192.0f);cb=max(min(cb,8191.0f),-8192.0f);
    ushort crp=pack_chroma(cr),cbp=pack_chroma(cb);
    if(a.area_flag==1) {
        float a16=float(int(crp)-0x8000),b16=float(int(cbp)-0x8000);
        float d0=area_edge(a16,b16,a,0,1,4,6),d1=area_edge(a16,b16,a,0,1,5,7);
        float d2=area_edge(a16,b16,a,2,3,8,10),d3=area_edge(a16,b16,a,2,3,9,11);
        float m=min(min(d0,d1),min(d2,d3));
        crp=area_out(int(crp),m*a.area[12]);cbp=area_out(int(cbp),m*a.area[13]);
    }
    int ys=int(short(yv));
    int lo=int(uint(a.w9)-uint(ys)),hiw=int(uint(ys)-uint(a.w12));
    int m=lo>0?int(uint(lo)*uint(a.w8)):int(uint(hiw)*uint(a.w11));
    int f=clamp(0xff-(m>>12),0,0xff);if(!(lo>0||hiw>0))f=0xff;
    yo[o]=ys>0?yv:ushort(0);cro[o]=scale_chroma(crp,f);cbo[o]=scale_chroma(cbp,f);
}
kernel void look_front(device const ushort *rgb [[buffer(0)]],device const float *table [[buffer(1)]],device const short *gamma [[buffer(2)]],device const float *atan_tab [[buffer(3)]],device ushort *yo [[buffer(4)]],device ushort *cro [[buffer(5)]],device ushort *cbo [[buffer(6)]],constant FrontArgs &a [[buffer(7)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    ycc_front(look_encode(rgb,table,gamma,atan_tab,p,a.width,a.thresholds),a,yo,cro,cbo,p.y*a.width+p.x);
}
kernel void encoded_front(device const short *enc [[buffer(0)]],device ushort *yo [[buffer(1)]],device ushort *cro [[buffer(2)]],device ushort *cbo [[buffer(3)]],constant FrontArgs &a [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    uint o=p.y*a.width+p.x,i=o*3;ycc_front(int3(enc[i],enc[i+1],enc[i+2]),a,yo,cro,cbo,o);
}

// GeoCorr + SSCS + the look front in one pass, without the interleaved RGB
// image between them. The front's hue blur reads its row at x +- 2, so a
// threadgroup computes GeoCorr/SSCS for its 64 x 4 pixels plus two columns on
// each side (68 x 4, 6% more than once per pixel) into threadgroup memory
// and runs the front from there. Each value is computed exactly as by
// geo_sscs_apply and look_front.
#define GF_W 64
#define GF_H 4
#define GF_TW (GF_W+4)
struct GeoFrontArgs { GeoSscsArgs gs; FrontArgs f; };
kernel void geo_front(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],
                      device const float *tr [[buffer(3)]],device const float *tg [[buffer(4)]],device const float *tb [[buffer(5)]],
                      device const float *table [[buffer(6)]],device const short *gamma [[buffer(7)]],device const float *atan_tab [[buffer(8)]],
                      device ushort *yo [[buffer(9)]],device ushort *cro [[buffer(10)]],device ushort *cbo [[buffer(11)]],
                      constant GeoFrontArgs &a [[buffer(12)]],uint2 grp [[threadgroup_position_in_grid]],
                      uint2 lt [[thread_position_in_threadgroup]],uint2 tsz [[threads_per_threadgroup]]) {
    threadgroup ushort T[GF_TW*GF_H*3];
    int w=int(a.f.width),h=int(a.f.height),X0=int(grp.x)*GF_W,Y0=int(grp.y)*GF_H;
    uint tid=lt.y*tsz.x+lt.x,nt=tsz.x*tsz.y;
    for(uint k=tid;k<GF_TW*GF_H;k+=nt) {
        int gx=X0-2+int(k%GF_TW),gy=Y0+int(k/GF_TW);
        ushort r0=0,g0=0,b0=0;
        if(gx>=0&&gx<w&&gy<h) {
            float px=float(a.gs.geo.region_x)+float(gx),py=float(a.gs.geo.region_y)+float(gy);
            float dx=px-a.gs.geo.cx,dy=py-a.gs.geo.cy;
            float radius=sqrt(fma(dy,dy,dx*dx)),t=(a.gs.geo.s2*radius)/a.gs.geo.step;
            int i0=int(floor(max(t,0.0f)));bool inside=i0<int(a.gs.geo.n)-1;
            int i=clamp(i0,0,int(a.gs.geo.n)-2);float frac=t-float(i);
            r0=geo_channel(r,tr,a.gs.geo,px,py,i,frac,inside,0);
            g0=geo_channel(g,tg,a.gs.geo,px,py,i,frac,inside,1);
            b0=geo_channel(b,tb,a.gs.geo,px,py,i,frac,inside,2);
            sscs_pixel(r0,g0,b0,a.gs.sscs);
        }
        T[3*k]=r0;T[3*k+1]=g0;T[3*k+2]=b0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint k=tid;k<GF_W*GF_H;k+=nt) {
        int lx=int(k%GF_W),ly=int(k/GF_W),px=X0+lx,py=Y0+ly;
        if(px>=w||py>=h)continue;
        int3 e=look_encode_at((threadgroup const ushort *)T,uint(ly*GF_TW+lx+2)*3u,uint(px),uint(w),table,gamma,atan_tab,a.f.thresholds);
        ycc_front(e,a.f,yo,cro,cbo,uint(py)*uint(w)+uint(px));
    }
}
