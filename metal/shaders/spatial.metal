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
// Vertical pass, Sharpness, then YGamma on every pixel.
kernel void sharp_apply(device const ushort *y [[buffer(0)]],device const float *hb [[buffer(1)]],device const short *lut [[buffer(2)]],device ushort *yg [[buffer(3)]],constant SpatialArgs &a [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    uint w=a.width,h=a.height;if(p.x>=w||p.y>=h)return;
    uint i=p.y*w+p.x;ushort v=y[i];
    if(p.x>=2&&p.x<w-2&&p.y>=2&&p.y<h-2) {
        uint wr=w-4,hr=h-4,xx=p.x-2,yy=p.y-2;float cab=0.0f;
        for(int o=-3;o<=3;o++){int sy=int(yy)+o;if(sy>=0&&sy<int(hr))cab=fma(hb[uint(sy)*wr+xx],sharp_w[3-o],cab);}
        float l=float(4*int(v)+int(y[i-w])+int(y[i+w])+int(y[i-1])+int(y[i+1]));
        float s=float(v),s7=fma(l,0.125f,cab*-0.000244140625f),s16=fma(-l,0.125f,s),d=fma(s7,a.sharp_s9,s16);
        float e=d>a.sharp_thr?a.sharp_s0*d:(d< -a.sharp_thr?a.sharp_s1*d:0.0f);
        v=ushort(int(trunc(clamp(fma(a.sharp_k,e,s),0.0f,32767.0f))));
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
    long m=min(min(long(g1),long(g2)),long(g3));
    return (a.s[24]*0.00048828125f)*float(m);
}
inline ushort div_clip(int v) { return ushort(clamp(v/10000,0,0x3fff)); }
// Spica on the gamma-corrected Y, then YCC2RGB.
kernel void spica_ycc(device const ushort *yg [[buffer(0)]],device const ushort *cr [[buffer(1)]],device const ushort *cb [[buffer(2)]],device const int *patterns [[buffer(3)]],device const float *coefs [[buffer(4)]],device ushort *ro [[buffer(5)]],device ushort *go [[buffer(6)]],device ushort *bo [[buffer(7)]],constant SpatialArgs &a [[buffer(8)]],uint2 p [[thread_position_in_grid]]) {
    uint w=a.width,h=a.height;if(p.x>=w||p.y>=h)return;
    uint i=p.y*w+p.x;int yc=int(yg[i]),yv=yc;
    if(p.x>=3&&p.x<w-3&&p.y>=3&&p.y<h-3) {
        int vals[9],mn=0x7fffffff,mx=-0x7fffffff-1;
        for(int k=0;k<9;k++){int v=int(yg[uint(int(i)+spica_cross[k].y*int(w)+spica_cross[k].x)]);vals[k]=v;mn=min(mn,v);mx=max(mx,v);}
        int rng=mx-mn,hf=rng/2,bits=0;
        for(int k=0;k<9;k++)if(hf<=vals[k]-mn)bits|=1<<k;
        int low=bits&0xff;uint idx=uint((bits&0x100)?(~low)&0xff:low);
        if(uint(rng)<a.spica_thr)idx=0;
        int ci=patterns[idx*3],dxs=patterns[idx*3+1],dys=patterns[idx*3+2];
        device const float *c=coefs+ci*26+1;float acc=0.0f;
        for(int t=0;t<25;t++){
            int ry=dys<0?-spica_taps[t].y:spica_taps[t].y,rx=dxs<0?-spica_taps[t].x:spica_taps[t].x;
            float val=float(yg[uint(int(i)+ry*int(w)+rx)]);acc=t==0?c[t]*val:fma(c[t],val,acc);
        }
        long tv;
        if(a.spica_full==0)tv=long(acc)>>9;
        else {
            float d=(acc-float(yc<<9))*a.spica_s,mid=float((mn+mx)/2);
            float g=spica_gain(d*0.0078125f,float(rng),mid,a);
            float m=0.001953125f*(g*0.001953125f);
            tv=long(d*m)+long(yc);
        }
        tv=clamp(tv,0L,long(0x3fff));
        float ycv=float(yc),o=fma(a.spica_k,float(tv)-ycv,ycv);
        yv=int(trunc(clamp(o,0.0f,32767.0f)));
    }
    int crv=int(cr[i])-0x8000,cbv=int(cb[i])-0x8000,yy=yv*10000;
    ro[i]=div_clip(crv*0x36c4+yy);go[i]=div_clip(crv*-0x1be5+yy+cbv*-0xd71);bo[i]=div_clip(cbv*0x4538+yy);
}
