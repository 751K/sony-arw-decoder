// ZcTaskSharpness, ZcTaskYGamma, ZcTaskSpica and ZcTaskYCC2RGB on the whole
// image. Sharpness writes [2, W-2) x [2, H-2) and its blur only sees that
// region; Spica writes [3, W-3) x [3, H-3). Other pixels pass through.
struct SpatialArgs {
    uint width,height;
    float sharp_thr,sharp_s0,sharp_s1,sharp_s9,sharp_k;
    float yg_s0,yg_s1,yg_s2,yg_s3;   // YGamma offset, gain, knee and 1/range
    uint spica_thr; int spica_full,spica_gain_on; float spica_s,spica_k;
    float s[25];
};
constant float sharp_w[7]={1.0f,6.0f,15.0f,20.0f,15.0f,6.0f,1.0f};
constant int2 spica_cross[9]={int2(0,-2),int2(0,-1),int2(-2,0),int2(-1,0),int2(0,0),int2(1,0),int2(2,0),int2(0,1),int2(0,2)};
constant int2 spica_taps[25]={int2(0,-3),int2(-1,-2),int2(0,-2),int2(1,-2),int2(-2,-1),int2(-1,-1),int2(0,-1),int2(1,-1),int2(2,-1),
    int2(-3,0),int2(-2,0),int2(-1,0),int2(0,0),int2(1,0),int2(2,0),int2(3,0),
    int2(-2,1),int2(-1,1),int2(0,1),int2(1,1),int2(2,1),int2(-1,2),int2(0,2),int2(1,2),int2(0,3)};

// Horizontal [1 6 15 20 15 6 1] pass over the (W-4) x (H-4) task region.
kernel void sharp_hblur(device const ushort *y [[buffer(0)]],device float *hb [[buffer(1)]],constant SpatialArgs &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
    uint wr=a.width-4,hr=a.height-4;if(p.x>=wr||p.y>=hr)return;
    device const ushort *row=y+(p.y+2)*a.width+2;float acc=0.0f;
    for(int o=-3;o<=3;o++){int sx=int(p.x)+o;if(sx>=0&&sx<int(wr))acc=fma(float(row[sx]),sharp_w[3-o],acc);}
    hb[p.y*wr+p.x]=acc;
}
inline ushort ygamma(ushort v,device const short *lut,constant SpatialArgs &a) {
    float x=float(ushort(lut[0x8000+uint(v)]));
    float s7=(x-a.yg_s2)*512.0f,s6=a.yg_s2<x?s7:0.0f;
    s6=fma(s6,a.yg_s3,-a.yg_s0);s6=fma(s6,a.yg_s1,a.yg_s0);
    return ushort(int(trunc(clamp(s6,0.0f,16383.0f))));
}
// Sharpness from the pixel, its vertical blur and its 4*v + 4-neighbour sum.
inline ushort sharp_value(ushort v,float cab,int lap,constant SpatialArgs &a) {
    float l=float(lap);
    float s=float(v),s7=fma(l,0.125f,cab*-0.000244140625f),s16=fma(-l,0.125f,s),d=fma(s7,a.sharp_s9,s16);
    float e=d>a.sharp_thr?a.sharp_s0*d:(d< -a.sharp_thr?a.sharp_s1*d:0.0f);
    return ushort(int(trunc(clamp(fma(a.sharp_k,e,s),0.0f,32767.0f))));
}
// Vertical pass, Sharpness, then YGamma on every pixel.
kernel void sharp_apply(device const ushort *y [[buffer(0)]],device const float *hb [[buffer(1)]],device const short *lut [[buffer(2)]],device ushort *yg [[buffer(3)]],constant SpatialArgs &a [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    uint w=a.width,h=a.height;if(p.x>=w||p.y>=h)return;
    uint i=p.y*w+p.x;ushort v=y[i];
    if(p.x>=2&&p.x<w-2&&p.y>=2&&p.y<h-2) {
        uint wr=w-4,hr=h-4,xx=p.x-2,yy=p.y-2;float cab=0.0f;
        for(int o=-3;o<=3;o++){int sy=int(yy)+o;if(sy>=0&&sy<int(hr))cab=fma(hb[uint(sy)*wr+xx],sharp_w[3-o],cab);}
        v=sharp_value(v,cab,4*int(v)+int(y[i-w])+int(y[i+w])+int(y[i-1])+int(y[i+1]),a);
    }
    yg[i]=ygamma(v,lut,a);
}
inline float spica_side(float p,constant float *s) {
    float a=s[0],a2=s[1],bb=s[2],b2=s[3],y0=s[4],y1=s[5],r=y0;
    if(p>=a&&p<a2)r=((y0-y1)*(a2-p))/(a2-a)+y1;
    if(p>=a2)r=p>=bb?(p<b2?((y1-y0)*(b2-p))/(b2-bb)+y0:y0):y1;
    return r;
}
inline float spica_curve(float p,constant float *s) {
    float a=s[0],a2=s[1],bb=s[2],b2=s[3],y0=s[4],y1=s[5],r=y0;bool lo=p>=a;
    if(lo&&p<a2)r=((y0-y1)*(a2-p))/(a2-a)+y1;
    if(lo&&p>=a2)r=p>=bb?(p<b2?((y1-y0)*(b2-p))/(b2-bb)+y0:y0):y1;
    return r;
}
inline float spica_gain(float d,float rng,float mid,constant SpatialArgs &a) {
    if(a.spica_gain_on==0)return 512.0f;
    float g1=d>0?spica_side(d,a.s):spica_side(-d,a.s+6);
    float g2=spica_curve(rng,a.s+12),g3=spica_curve(mid,a.s+18);
    // 64-bit integers are emulated on the GPU. Each curve value lies between
    // its segment ends y0 and y1 (s[4..5] of each 6-value set), so when those
    // are well inside the int range (uniform over the dispatch) 32-bit
    // truncation gives the same minimum.
    float lim=1073741824.0f;
    bool small=true;
    for(int k=0;k<4;k++)small=small&&abs(a.s[6*k+4])<lim&&abs(a.s[6*k+5])<lim;
    if(small)return (a.s[24]*0.00048828125f)*float(min(min(int(g1),int(g2)),int(g3)));
    long m=min(min(long(g1),long(g2)),long(g3));
    return (a.s[24]*0.00048828125f)*float(m);
}
// long(acc)>>9 and long(dm)+yc clamped to [0, 0x3fff], in 32-bit integers:
// inside the int range the truncation is the same, outside it the clamp
// gives the same end value.
inline int spica_tv_shift(float acc) {
    if(acc>=2147483648.0f)return 0x3fff;
    if(acc< -2147483648.0f)return 0;
    return clamp(int(acc)>>9,0,0x3fff);
}
inline int spica_tv_add(float dm,int yc) {
    if(dm>=1073741824.0f)return 0x3fff;
    if(dm<=-1073741824.0f)return 0;
    return clamp(int(dm)+yc,0,0x3fff);
}
inline ushort div_clip(int v) { return ushort(clamp(v/10000,0,0x3fff)); }
// Spica at yg[i] (row stride `stride`): device memory or a threadgroup tile.
template<typename P>
inline int spica_value(P yg,int i,int stride,int yc,device const int *patterns,device const float *coefs,constant SpatialArgs &a) {
    int vals[9],mn=0x7fffffff,mx=-0x7fffffff-1;
    for(int k=0;k<9;k++){int v=int(yg[i+spica_cross[k].y*stride+spica_cross[k].x]);vals[k]=v;mn=min(mn,v);mx=max(mx,v);}
    int rng=mx-mn,hf=rng/2,bits=0;
    for(int k=0;k<9;k++)if(hf<=vals[k]-mn)bits|=1<<k;
    int low=bits&0xff;uint idx=uint((bits&0x100)?(~low)&0xff:low);
    if(uint(rng)<a.spica_thr)idx=0;
    int ci=patterns[idx*3],dxs=patterns[idx*3+1],dys=patterns[idx*3+2];
    device const float *c=coefs+ci*26+1;float acc=0.0f;
    for(int t=0;t<25;t++){
        int ry=dys<0?-spica_taps[t].y:spica_taps[t].y,rx=dxs<0?-spica_taps[t].x:spica_taps[t].x;
        float val=float(yg[i+ry*stride+rx]);acc=t==0?c[t]*val:fma(c[t],val,acc);
    }
    int tv;
    if(a.spica_full==0)tv=spica_tv_shift(acc);
    else {
        float d=(acc-float(yc<<9))*a.spica_s,mid=float((mn+mx)/2);
        float g=spica_gain(d*0.0078125f,float(rng),mid,a);
        float m=0.001953125f*(g*0.001953125f);
        tv=spica_tv_add(d*m,yc);
    }
    float ycv=float(yc),o=fma(a.spica_k,float(tv)-ycv,ycv);
    return int(trunc(clamp(o,0.0f,32767.0f)));
}
// Spica on the gamma-corrected Y, then YCC2RGB.
kernel void spica_ycc(device const ushort *yg [[buffer(0)]],device const ushort *cr [[buffer(1)]],device const ushort *cb [[buffer(2)]],device const int *patterns [[buffer(3)]],device const float *coefs [[buffer(4)]],device ushort *ro [[buffer(5)]],device ushort *go [[buffer(6)]],device ushort *bo [[buffer(7)]],constant SpatialArgs &a [[buffer(8)]],uint2 p [[thread_position_in_grid]]) {
    uint w=a.width,h=a.height;if(p.x>=w||p.y>=h)return;
    uint i=p.y*w+p.x;int yc=int(yg[i]),yv=yc;
    if(p.x>=3&&p.x<w-3&&p.y>=3&&p.y<h-3) {
        yv=spica_value(yg,int(i),int(w),yc,patterns,coefs,a);
    }
    int crv=int(cr[i])-0x8000,cbv=int(cb[i])-0x8000,yy=yv*10000;
    ro[i]=div_clip(crv*0x36c4+yy);go[i]=div_clip(crv*-0x1be5+yy+cbv*-0xd71);bo[i]=div_clip(cbv*0x4538+yy);
}

// Sharpness, YGamma, Spica and YCC2RGB in one pass on 32 x 32 output tiles,
// with the intermediates in threadgroup memory instead of the hb and yg planes.
// Spica reads yg at +-3, yg needs the vertical blur at +-3 rows, and the
// horizontal blur needs Y at +-3 columns, so a tile loads Y with a 6-pixel
// border. Every value is computed with the same operations, in the same
// order, as sharp_hblur, sharp_apply and spica_ycc.
#define SP_T 32
#define SP_YW (SP_T+12)              // Y tile: image (X0-6.., Y0-6..)
#define SP_HW (SP_T+6)               // hb tile: rows Y0-6.., columns X0-3..; yg tile: (X0-3.., Y0-3..)
kernel void spatial_fused(device const ushort *y [[buffer(0)]],device const ushort *cr [[buffer(1)]],device const ushort *cb [[buffer(2)]],
                          device const short *lut [[buffer(3)]],device const int *patterns [[buffer(4)]],device const float *coefs [[buffer(5)]],
                          device ushort *ro [[buffer(6)]],device ushort *go [[buffer(7)]],device ushort *bo [[buffer(8)]],
                          constant SpatialArgs &a [[buffer(9)]],uint2 tg [[threadgroup_position_in_grid]],
                          uint2 lt [[thread_position_in_threadgroup]],uint2 tsz [[threads_per_threadgroup]]) {
    threadgroup ushort Y[SP_YW*SP_YW];
    threadgroup float HB[SP_YW*SP_HW];
    threadgroup ushort YG[SP_HW*SP_HW];
    int w=int(a.width),h=int(a.height),X0=int(tg.x)*SP_T,Y0=int(tg.y)*SP_T;
    uint tid=lt.y*tsz.x+lt.x,nt=tsz.x*tsz.y;
    for(uint k=tid;k<SP_YW*SP_YW;k+=nt){
        int gx=X0-6+int(k%SP_YW),gy=Y0-6+int(k/SP_YW);
        Y[k]=(gx>=0&&gx<w&&gy>=0&&gy<h)?y[uint(gy)*uint(w)+uint(gx)]:ushort(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Horizontal blur at image (gx, gy) = sharp_hblur's hb[gy-2][gx-2].
    for(uint k=tid;k<SP_YW*SP_HW;k+=nt){
        int r=int(k/SP_HW),c=int(k%SP_HW),gx=X0-3+c,gy=Y0-6+r;float acc=0.0f;
        if(gx>=2&&gx<w-2&&gy>=2&&gy<h-2)
            for(int o=-3;o<=3;o++){int sx=gx+o;if(sx>=2&&sx<w-2)acc=fma(float(Y[r*SP_YW+c+3+o]),sharp_w[3-o],acc);}
        HB[k]=acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Vertical pass, Sharpness and YGamma: sharp_apply's yg.
    for(uint k=tid;k<SP_HW*SP_HW;k+=nt){
        int r=int(k/SP_HW),c=int(k%SP_HW),gx=X0-3+c,gy=Y0-3+r;
        if(gx<0||gx>=w||gy<0||gy>=h){YG[k]=0;continue;}
        int yi=(r+3)*SP_YW+c+3;ushort v=Y[yi];
        if(gx>=2&&gx<w-2&&gy>=2&&gy<h-2){
            float cab=0.0f;
            for(int o=-3;o<=3;o++){int sy=gy+o;if(sy>=2&&sy<h-2)cab=fma(HB[(r+3+o)*SP_HW+c],sharp_w[3-o],cab);}
            v=sharp_value(v,cab,4*int(v)+int(Y[yi-SP_YW])+int(Y[yi+SP_YW])+int(Y[yi-1])+int(Y[yi+1]),a);
        }
        YG[k]=ygamma(v,lut,a);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Spica and YCC2RGB: spica_ycc on the yg tile.
    for(uint k=tid;k<SP_T*SP_T;k+=nt){
        int px=X0+int(k%SP_T),py=Y0+int(k/SP_T);
        if(px>=w||py>=h)continue;
        int li=(py-Y0+3)*SP_HW+(px-X0+3);
        uint i=uint(py)*uint(w)+uint(px);int yc=int(YG[li]),yv=yc;
        if(px>=3&&px<w-3&&py>=3&&py<h-3) {
            yv=spica_value((threadgroup const ushort *)YG,li,SP_HW,yc,patterns,coefs,a);
        }
        int crv=int(cr[i])-0x8000,cbv=int(cb[i])-0x8000,yy=yv*10000;
        ro[i]=div_clip(crv*0x36c4+yy);go[i]=div_clip(crv*-0x1be5+yy+cbv*-0xd71);bo[i]=div_clip(cbv*0x4538+yy);
    }
}
