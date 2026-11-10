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
inline float look_blur(device const ushort *rgb,uint x,uint y,uint w,uint channel) {
    uint i=(y*w+x)*3+channel;float a,b;
    if(x==0)return fma(float(rgb[i+3]),8.0f,fma(float(rgb[i]),23.0f,float(rgb[i+6])))*0.03125f;
    if(x==1)return fma(float(rgb[i+3]),8.0f,fma(float(rgb[i]),14.0f,fma(float(rgb[i-3]),9.0f,float(rgb[i+6]))))*0.03125f;
    if(x==w-2){a=float(rgb[i+3])+float(rgb[i-3]);b=fma(float(rgb[i]),14.0f,float(rgb[i-6]))+float(rgb[i+3]);return fma(a,8.0f,b)*0.03125f;}
    if(x==w-1)return fma(float(rgb[i]),23.0f,fma(float(rgb[i-3]),8.0f,float(rgb[i-6])))*0.03125f;
    a=float(rgb[i+3])+float(rgb[i-3]);b=fma(float(rgb[i]),14.0f,float(rgb[i-6]))+float(rgb[i+6]);return fma(a,8.0f,b)*0.03125f;
}
inline int3 look_encode(device const ushort *rgb,device const float *table,device const short *gamma,device const float *atan_tab,uint2 p,uint width,constant float *thresholds) {
    uint i=(p.y*width+p.x)*3;float r=rgb[i],g=rgb[i+1],b=rgb[i+2];
    float br=look_blur(rgb,p.x,p.y,width,0),bg=look_blur(rgb,p.x,p.y,width,1),bb=look_blur(rgb,p.x,p.y,width,2);
    // The blur is an exact multiple of 1/32 for uint16 input. Summing its
    // integer numerator preserves the reference's double-precision luma.
    int luma_num=int(br*32.0f)*109+int(bg*32.0f)*366+int(bb*32.0f)*37;
    float lum=float(luma_num)*0.00006103515625f;
    float aa=br-lum,bc=bb-lum,fa=abs(aa),fb=abs(bc);bool big=fa>fb,ap=!(aa<0),bp=!(bc<0);
    float offh=big?(ap?(bp?-256.0f:256.0f):(bp?768.0f:-768.0f)):(ap?(bp?0.0f:-512.0f):(bp?-1024.0f:512.0f));
    float num=big?fb:fa,den=big?(aa==0?1.0f:fa):(bc==0?1.0f:fb);
    // Differences are multiples of 1/16384. Integer division reproduces the
    // reference's float64 ratio and round-to-nearest at hue-bin boundaries.
    long ni=long(num*16384.0f),di=long(den*16384.0f);
    int index=clamp(int((ni*1024L+di)/(di*2L)),0,511);
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
    long sum=long(enc.x)*long(a.c0)+long(enc.y)*long(a.c1)+long(enc.z)*long(a.c2);
    ushort yv=ushort((uint(sum)>>13)&0xffffu);
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
