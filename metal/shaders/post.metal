// ZcTaskEffect (B&W / Sepia), ZcTaskToneCurve and ZcTaskShadeCorrection,
// pointwise and in that order.
//
// Effect's luminance and Shade's gain and Y / gain are float64 in the
// reference; Metal has no double. They are evaluated here in double-float
// (hi + lo) arithmetic, accurate to about 2^-44. The float64 result can then
// only disagree where a value lies within that distance of a rounding or
// truncation boundary; such pixels are appended to `fix` and the host
// recomputes them with the reference code.
struct PostArgs {
    uint width,height,capacity;
    int effect;              // 0 off, 1 B&W, 2 Sepia
    int conv;                // Effect colour conversion: 0 none, 1 on
    int m_fwd[9],m_inv[9];
    float t_r,t_b;
    float c_hi[3],c_lo[3];   // 0.2126, 0.7152, 0.0722 split into float pairs
    int tone,shade;
    int cw,ch;               // Shade centre dimensions
    float g_hi[11],g_lo[11]; // gain coefficients
    float d_hi[10],d_lo[10]; // coeffs[i+1] - coeffs[i] in float64
    uint diff_zero,lo_one;   // per-index bits: difference exactly 0, coefficient exactly 1
    int c10_one;
};
struct dd { float hi,lo; };
inline dd two_sum(float a,float b) { float s=a+b,bb=s-a;return {s,(a-(s-bb))+(b-bb)}; }
inline dd fast_two_sum(float a,float b) { float s=a+b;return {s,b-(s-a)}; }
inline dd dd_from_long(long v) { float h=float(v);return fast_two_sum(h,float(v-long(h))); }
inline dd dd_add(dd x,dd y) {
    dd s=two_sum(x.hi,y.hi),t=two_sum(x.lo,y.lo);
    s=fast_two_sum(s.hi,s.lo+t.hi);return fast_two_sum(s.hi,s.lo+t.lo);
}
inline dd dd_neg(dd x) { return {-x.hi,-x.lo}; }
inline dd dd_mul_f(dd x,float f) { float p=x.hi*f,e=fma(x.hi,f,-p);e=fma(x.lo,f,e);return fast_two_sum(p,e); }
inline dd dd_mul(dd x,dd y) { float p=x.hi*y.hi,e=fma(x.hi,y.hi,-p);e=fma(x.hi,y.lo,fma(x.lo,y.hi,e));return fast_two_sum(p,e); }
inline dd dd_div(dd x,dd y) {
    float q1=x.hi/y.hi;dd r=dd_add(x,dd_neg(dd_mul_f(y,q1)));
    float q2=r.hi/y.hi;r=dd_add(r,dd_neg(dd_mul_f(y,q2)));
    float q3=r.hi/y.hi;return dd_add(fast_two_sum(q1,q2),dd{q3,0.0f});
}
inline dd dd_sqrt(dd x) {
    if(x.hi<=0)return {0.0f,0.0f};
    float s=sqrt(x.hi),p=s*s,e=fma(s,s,-p);
    dd r=dd_add(x,dd_neg(dd{p,e}));return fast_two_sum(s,r.hi/(2.0f*s));
}
// floor(x), or -1 when x is within `margin` of an integer.
inline long dd_floor_checked(dd x,float margin) {
    float k=rint(x.hi),d=(x.hi-k)+x.lo;
    if(abs(d)<margin)return -1;
    return long(k)-(d<0?1:0);
}
inline void post_conv(thread int *v,device const short *dec,constant int *m,device const short *enc) {
    long s0=dec[min(v[0],0x3fff)],s1=dec[min(v[1],0x3fff)],s2=dec[min(v[2],0x3fff)];
    for(int c=0;c<3;c++){long o=(long(m[3*c])*s0+long(m[3*c+1])*s1+long(m[3*c+2])*s2)>>13;v[c]=int(ushort(enc[clamp(o,0L,0x3fffL)]));}
}
// float32 rounding of the luminance R*0.2126 + G*0.7152 + B*0.0722; false when ambiguous.
inline bool effect_luma(int r,int g,int b,constant PostArgs &a,thread float &out) {
    dd d=dd_mul_f(dd{a.c_hi[0],a.c_lo[0]},float(r));
    d=dd_add(d,dd_mul_f(dd{a.c_hi[1],a.c_lo[1]},float(g)));
    d=dd_add(d,dd_mul_f(dd{a.c_hi[2],a.c_lo[2]},float(b)));
    out=d.hi;if(d.hi==0)return true;
    int e;float m=frexp(d.hi,e),up=ldexp(1.0f,e-24),down=m==0.5f?up*0.5f:up;
    float dist=d.lo>=0?abs(d.lo-up*0.5f):abs(-d.lo-down*0.5f);
    return dist>=d.hi*0x1p-40f;
}
inline bool effect(thread int *v,device const short *dec_in,device const short *enc_in,device const short *dec_out,device const short *enc_out,constant PostArgs &a) {
    if(a.conv)post_conv(v,dec_in,a.m_fwd,enc_in);
    int s=v[0]*0x991+v[1]*0x12c9+v[2]*0x3a6;
    if(a.effect==1){int y=min(s>>13,0x3fff);v[0]=v[1]=v[2]=y;return true;}
    float y13=float(s>>13),lo=y13*0.00074404763f,hi=fma(y13,as_type<float>(0xba3a2e8cu),as_type<float>(0x4139745du));
    float wr=(s>>19)>0x14?a.t_r:lo*a.t_r,wb=(s>>19)>0x14?a.t_b:lo*a.t_b;
    if(!(s<0x7482000)){wr=hi*a.t_r;wb=hi*a.t_b;}
    float l;if(!effect_luma(v[0],v[1],v[2],a,l))return false;
    l=max(min(l,16383.0f),0.0f);
    long ro=long(trunc(wr+l)),bo=long(trunc(wb+l));
    long go=long(trunc(l))-((long(trunc(wr))*0x104c+long(trunc(wb))*0x636)>>13);
    v[0]=int(clamp(ro,0L,0x3fffL));v[1]=int(clamp(go,0L,0x3fffL));v[2]=int(clamp(bo,0L,0x3fffL));
    if(a.conv)post_conv(v,dec_out,a.m_inv,enc_out);
    return true;
}
inline void tone(thread int *v,device const ushort *luts) {
    int R=v[0],G=v[1],B=v[2];
    int Y=(R*0x991+G*0x12c9+B*0x3a6)>>11,C1=(R*-0x567+G*-0xa9a+B*0x1000)>>11,C2=(G*-0xd66+R*0x1000+B*-0x29a)>>11;
    int y8=int(luts[3*0x10000+clamp(Y,0,0xffff)])*0x2000;
    int c[3]={(y8+C2*0x2cdd)>>13,(y8+C1*-0xb03+C2*-0x16da)>>13,(y8+C1*0x38b4+C2)>>13};
    for(int k=0;k<3;k++)v[k]=(int(luts[k*0x10000+clamp(c[k],0,0xffff)])>>2)&0x3fff;
}
inline bool shade(thread int *v,uint x,uint y,constant PostArgs &a) {
    long dx=long(a.cw)-2*long(x),dy=long(a.ch)-2*long(y);
    dd ratio=dd_div(dd_from_long(dx*dx+dy*dy),dd_from_long(long(a.cw)*a.cw+long(a.ch)*a.ch));
    dd t=dd_mul_f(dd_sqrt(ratio),10.0f);
    long idx=dd_floor_checked(t,0x1p-36f);if(idx<0)return false;
    dd g;bool keep;
    if(idx>=10){g={a.g_hi[10],a.g_lo[10]};keep=a.c10_one!=0;}
    else if(a.diff_zero>>idx&1u){g={a.g_hi[idx],a.g_lo[idx]};keep=(a.lo_one>>idx&1u)!=0;}
    else {
        dd frac=dd_add(t,dd{-float(idx),0.0f});
        g=dd_add(dd{a.g_hi[idx],a.g_lo[idx]},dd_mul(frac,dd{a.d_hi[idx],a.d_lo[idx]}));
        dd off=dd_add(g,dd{-1.0f,0.0f});
        if(abs(off.hi)<0x1p-40f)return false;     // could be exactly 1.0 in float64
        keep=false;
    }
    if(keep)return true;
    int R=v[0],G=v[1],B=v[2];
    int Y=(R*0x991+G*0x12c9+B*0x3a6)>>11;long yq=0;
    if(Y>0) {
        if(abs(g.hi)<0x1p-20f)return false;
        if(g.hi>0) {
            dd q=dd_div(dd_from_long(Y),g);
            if(q.hi>65536.5f)yq=0xffff;
            else {yq=dd_floor_checked(q,max(q.hi,1.0f)*0x1p-34f);if(yq<0)return false;}
        }
    }
    yq=clamp(yq,0L,0xffffL);
    int C1=(R*-0x567+G*-0xa9a+B*0x1000)>>11,C2=(G*-0xd66+R*0x1000+B*-0x29a)>>11;long y8=yq*0x2000;
    v[0]=int(clamp((y8+C2*0x2cdd)>>15,0L,0x3fffL));
    v[1]=int(clamp((y8+C1*-0xb03+C2*-0x16da)>>15,0L,0x3fffL));
    v[2]=int(clamp((y8+C1*0x38b4+C2)>>15,0L,0x3fffL));
    return true;
}
kernel void post_apply(device const ushort *ri [[buffer(0)]],device const ushort *gi [[buffer(1)]],device const ushort *bi [[buffer(2)]],
                       device const short *dec_in [[buffer(3)]],device const short *enc_in [[buffer(4)]],device const short *dec_out [[buffer(5)]],device const short *enc_out [[buffer(6)]],
                       device const ushort *luts [[buffer(7)]],device ushort *ro [[buffer(8)]],device ushort *go [[buffer(9)]],device ushort *bo [[buffer(10)]],
                       device uint *fix [[buffer(11)]],device atomic_uint *nfix [[buffer(12)]],constant PostArgs &a [[buffer(13)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    uint i=p.y*a.width+p.x;int v[3]={int(ri[i]),int(gi[i]),int(bi[i])};bool ok=true;
    if(a.effect)ok=effect(v,dec_in,enc_in,dec_out,enc_out,a);
    if(ok&&a.tone)tone(v,luts);
    if(ok&&a.shade)ok=shade(v,p.x,p.y,a);
    if(!ok){uint k=atomic_fetch_add_explicit(nfix,1u,memory_order_relaxed);if(k<a.capacity)fix[k]=i;}
    ro[i]=ushort(v[0]);go[i]=ushort(v[1]);bo[i]=ushort(v[2]);
}

// Planar R, G, B -> interleaved RGB (the render's final layout).
kernel void interleave3(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],
                        device ushort *o [[buffer(3)]],constant uint2 &d [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=d.x||p.y>=d.y)return;
    uint i=p.y*d.x+p.x;o[3*i]=r[i];o[3*i+1]=g[i];o[3*i+2]=b[i];
}
