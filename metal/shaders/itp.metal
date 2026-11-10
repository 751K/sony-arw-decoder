#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct Args { uint width, height; int black[4]; float gains[4]; int scale; float blend_k, blend_lo, blend_hi, blend_half; float k1,k2,k3,k4,k5,k6; uint image_width,image_height,tile_x,tile_y,tile_width,tile_height,halo; };
inline float get(device const float *p, int x, int y, int w, int h) {
    return x < 0 || y < 0 || x >= w || y >= h ? 0.0f : p[y*w+x];
}
inline float weighted(device const float *p, int x, int y, int w, int h, int dx) {
#define P(a,b) get(p,x+(a),y+(b),w,h)
    float s = (P(0,-2)+P(1,-2))+(P(3,-2)+P(4,-2));
    float t = (P(-2,-1)+P(2,-1))+P(-2,1); s += t;
    t = ((P(2,1)+P(-2,2))+P(-1,2))+P(1,2); s += t; s += P(2,2);
    float u = (P(-1,-1)+P(2,-2))+(P(0,-1)+P(1,-1));
    float v = (P(0,0)+P(1,0))+P(3,0); u += v;
    v = ((P(4,0)+P(-1,1))+P(0,1))+P(1,1); u += v; u += P(0,2);
    s = fma(u,6.0f,s);
    return s + P(dx,0)*36.0f;
#undef P
}
kernel void scale_raw(device const ushort *raw [[buffer(0)]], device float *out [[buffer(1)]], constant Args &a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= a.width || p.y >= a.height) return;
    uint b = (p.x & 1) | ((p.y & 1) << 1), i = p.y*a.width+p.x;
    float k = (float(a.scale)*3.0517578e-05f)*a.gains[b];
    out[i] = (k*0.0625f)*float(int(raw[i])-a.black[b]);
}
kernel void differences(device const float *src [[buffer(0)]], device float *ah [[buffer(1)]], device float *av [[buffer(2)]], constant Args &a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
    uint x=p.x,y=p.y,w=a.width,h=a.height; if (x>=w || y>=h) return;
    uint i=y*w+x; float par=float((x^y)&1);
    ah[i] = (x>=2 && x+2<w) ? abs(fma(src[i],-2.0f,src[i-2]+src[i+2]) * -0.25f)*par : 0.0f;
    av[i] = (y>=2 && y+2<h) ? abs(fma(src[i],-2.0f,src[i-2*w]+src[i+2*w]) * -0.25f)*par : 0.0f;
}
kernel void candidates(device const float *src [[buffer(0)]], device float *out [[buffer(1)]], constant Args &a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    int x=int(p.x), y=int(p.y), w=int(a.width), h=int(a.height); if(x>=w || y>=h) return;
    float v=0.0f;
    bool valid_y = y>=2 && ((y&1) ? y<h-1 : y<h-2);
    bool valid_x = x>=2 && ((x&1) ? x<w-1 : x<w-2);
    if (valid_y && valid_x) {
        int dx = ((x^y)&1) ? 2 : 0;
        v=weighted(src,x,y,w,h,dx) * ((y&1) ? as_type<float>(0x3c3a2e8cu) : 0.03125f);
    }
    if (valid_y && x<2) v=src[y*w+x];
    int xend = w>4 ? 2+2*((w-3)/2) : 2;
    if (valid_y && xend+1<w && (x==xend || x==xend+1)) v=src[y*w+2+(x-xend)];
    out[y*w+x]=v;
}
kernel void weights(device const float *a0 [[buffer(0)]], device const float *b0 [[buffer(1)]], device float *dir [[buffer(2)]], device float *blend [[buffer(3)]], constant Args &a [[buffer(4)]], uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width || p.y>=a.height) return;
    uint i=p.y*a.width+p.x; float av=a0[i], bv=b0[i];
    float t1=max(av-1.0f,0.0f)+0.125f, t2=max(bv-1.0f,0.0f), den=(t1+0.125f)+t2;
    dir[i]=den>0.0f ? t1/den : 0.5f;
    float ratio=(a.blend_k+av*8.0f)/(a.blend_k+bv*8.0f);
    float inv=ratio>1.0f ? 1.0f/ratio : ratio;
    float v=a.blend_lo+(inv-a.blend_lo)*a.blend_half;
    blend[i]=a.blend_lo<=ratio ? min(v,1.0f) : a.blend_hi;
}
kernel void green_inputs(device const float *s [[buffer(0)]], device float *mosaic [[buffer(1)]], device float *mix_a [[buffer(2)]], device float *mix_b [[buffer(3)]], device float *tap_a [[buffer(4)]], device float *tap_b [[buffer(5)]], constant Args &a [[buffer(6)]], uint2 p [[thread_position_in_grid]]) {
    int x=int(p.x), y=int(p.y), w=int(a.width), h=int(a.height); if(x>=w || y>=h) return;
    int i=y*w+x; mosaic[i]=((x^y)&1) ? s[i] : 0.0f;
    float ma=0,mb=0,ta=0,tb=0;
    bool mx=x>=2 && ((x&1) ? x<w-1 : x<w-2);
    bool my=y>=2 && ((y&1) ? y<h-2 : y<h-3);
    if(mx && my) {
        if((x^y)&1) { ma=fma(s[i],6.0f,s[i+2]+s[i-2])*0.125f; mb=fma(s[i],6.0f,s[i+2*w]+s[i-2*w])*0.125f; }
        else { ma=(s[i-1]+s[i+1])*0.5f; mb=(s[i-w]+s[i+w])*0.5f; }
    }
    bool tx=x>=4 && ((x&1) ? x<w-3 : x<w-4);
    bool ty=y>=4 && ((y&1) ? y<h-3 : y<h-4);
    if(tx && ty) {
        constexpr float k1[9]={-0.0625f,0.0f,0.25f,0.0f,0.625f,0.0f,0.25f,0.0f,-0.0625f};
        constexpr float k2[9]={0.0f,-0.0556640625f,0.0f,0.5556640625f,0.0f,0.5556640625f,0.0f,-0.0556640625f,0.0f};
        bool odd=(x^y)&1; ta=(odd?k1[0]:k2[0])*get(s,x-4,y,w,h); tb=(odd?k1[0]:k2[0])*get(s,x,y-4,w,h);
        for(int j=1;j<9;j++) { float k=odd?k1[j]:k2[j]; ta=fma(k,get(s,x-4+j,y,w,h),ta); tb=fma(k,get(s,x,y-4+j,w,h),tb); }
    }
    mix_a[i]=ma; mix_b[i]=mb; tap_a[i]=ta; tap_b[i]=tb;
}
kernel void green_five(device const float *s [[buffer(0)]], device float *out [[buffer(1)]], constant Args &a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
    int x=int(p.x), y=int(p.y), w=int(a.width), h=int(a.height); if(x>=w || y>=h) return;
    bool vx=x>=2 && ((x&1) ? x<w-1 : x<w-2);
    bool vy=y>=2 && ((y&1) ? y<h-1 : y<h-2);
    float v=0;
    if(vx && vy) {
#define P(dx,dy) get(s,x+(dx),y+(dy),w,h)
        if(!((x^y)&1)) {
            float e0=P(1,-2),e1=P(-1,-2),e2=P(-2,-1),e3=P(2,-1),e4=P(-2,1),e5=P(2,1),e6=P(-1,2),e7=P(1,2);
            float sum=(((e0+e1)+(e2+e3))+((e4+e5)+e6))+e7;
            float c=(P(-1,0)+P(0,-1))+(P(1,0)+P(0,1));
            v=fma(c,0.3125f,sum*-0.03125f);
        } else {
            float corners=(P(2,-2)+P(-2,-2))+(P(-2,2)+P(2,2));
            float axial=(P(-2,0)+P(0,-2))+(P(2,0)+P(0,2));
            float diag=(P(1,-1)+P(-1,-1))+(P(-1,1)+P(1,1));
            v=corners*-0.015625f; v=fma(axial,-0.03125f,v); v=fma(diag,0.125f,v); v=fma(P(0,0),0.6875f,v);
        }
#undef P
    }
    out[y*w+x]=v;
}
kernel void average_pairs(device const float *ma [[buffer(0)]],device const float *mb [[buffer(1)]],device const float *ta [[buffer(2)]],device const float *tb [[buffer(3)]], device float *oa [[buffer(4)]],device float *ob [[buffer(5)]],constant Args &a [[buffer(6)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width || p.y>=a.height) return; uint i=p.y*a.width+p.x; oa[i]=(ta[i]+ma[i])*0.5f; ob[i]=(tb[i]+mb[i])*0.5f;
}
inline float err3(float c,float a,float b) { return abs(a-b)+abs(c-a)+abs((c-(a+a))+b); }
kernel void direction_errors(device const float *s [[buffer(0)]], device float *eh [[buffer(1)]],device float *ev [[buffer(2)]],device float *e1 [[buffer(3)]],device float *e2 [[buffer(4)]],constant Args &a [[buffer(5)]],uint2 p [[thread_position_in_grid]]) {
 int x=p.x,y=p.y,w=a.width,h=a.height;if(x>=w||y>=h)return;int i=y*w+x;float hh=0,vv=0,d1=0,d2=0;
 if(y>=2 && y<h-4) {if(x>=2&&x<w-4){hh=err3(s[i],s[i+2],s[i+4]);vv=err3(s[i],s[i+2*w],s[i+4*w]);d2=err3(s[i],s[i+w+1],s[i+2*w+2]);} if(x>=4&&x<w-2)d1=err3(s[i],s[i+w-1],s[i+2*w-2]);}
 eh[i]=hh;ev[i]=vv;e1[i]=d1;e2[i]=d2;
}
kernel void direction(device const float *eh [[buffer(0)]], device const float *ev [[buffer(1)]],device const float *e1 [[buffer(2)]],device const float *e2 [[buffer(3)]],device float *out [[buffer(4)]],constant Args &a [[buffer(5)]],uint2 p [[thread_position_in_grid]]) {
 int x=p.x,y=p.y,w=a.width,h=a.height;if(x>=w||y>=h)return; int i=y*w+x; float res=0;
 if(x>=4 && x<w-4 && y>=4 && y<h-4) {
#define V(z,dx,dy) z[(y+(dy))*w+x+(dx)]
 float sc=((x^y)&1)?0.0625f:0.03125f;
 float t1=V(e1,-1,-2),t2=V(e1,-2,-2),t3=V(e1,0,-2),t4=V(e1,-2,-1),t5=V(e1,-1,-1),t6=V(e1,0,-1),t7=V(e1,-2,0),t8=V(e1,-1,0),t9=V(e1,0,0);
 float tf=(t1+t2)+(t3+t4);float tt=((tf+4.0f*t5)+(((t6+t7)+t8)+t9));
 float f42=max(fma(0.25f,tt,-a.k1),a.k2);
 t1=V(e2,1,-2);t2=V(e2,0,-2);t3=V(e2,2,-2);t4=V(e2,0,-1);t5=V(e2,1,-1);t6=V(e2,2,-1);t7=V(e2,0,0);t8=V(e1,1,0);t9=V(e2,2,0);
 tf=(t1+t2)+(t3+t4);tt=((tf+4.0f*t5)+(((t6+t7)+t8)+t9));
 float f45=max(fma(0.25f,tt,-a.k1),a.k2), denom=(f45+f42)+0.01f;
 float am1=V(ev,-2,-3),a0=V(ev,-1,-3),ap1=V(ev,0,-3),ap2=V(ev,1,-3),ap3=V(ev,2,-3);
 float bm2=V(ev,-2,-2),b0=V(ev,0,-2),bp1=V(ev,1,-2),bp2=V(ev,2,-2);
 float cm2=V(ev,-2,-1),cm1=V(ev,-1,-1),c0=V(ev,0,-1),cp1=V(ev,1,-1),cp2=V(ev,2,-1);
 float t32=fma(a0,2.0f,am1);float d20=(b0+ap1)+c0,d22=(cm1+ap2)+cp1,d21=(ap3+t32);
 d21=bm2*9.0f+d21;d21=bp1*8.0f+d21;float d23=(bp2+cm2)+cp2;d21+=d23;d20=d20*6.0f+d21;d20=d22*2.0f+d20;
 float f45b=max(fma(sc,d20,-a.k1),a.k2);
 float pm1=V(eh,-3,-1),p0=V(eh,-2,-1),pp1=V(eh,-1,-1),qm1=V(eh,-3,-2),q0=V(eh,-2,-2),qp1=V(eh,-1,-2);
 float sm1=V(eh,-3,0),s0=V(eh,-2,0),sp1=V(eh,-1,0),tm1=V(eh,-3,1),t0=V(eh,-2,1),tp1=V(eh,-1,1),um1=V(eh,-3,2),u0=V(eh,-2,2),up1=V(eh,-1,2);
 float fq=fma(pm1,2.0f,(q0+qm1)+qp1);d21=p0*8.0f+fq;d21+=t0*8.0f;float d24=(s0+sm1)+sp1;d21=d24*6.0f+d21;d22=(tm1+pp1)+tp1;d23=(um1+u0)+up1;d21+=d23;d21=d22*2.0f+d21;
 float f43b=max(fma(sc,d21,-a.k1),a.k2);float num=(f43b+f45b)+0.01f;float quot=denom==0?1.0f:num/denom;float v=((a.k3-quot)+a.k4)*a.k5;res=v>a.k3?a.k3:max(v,a.k6);
#undef V
 } out[i]=res;
}
kernel void green_finish(device const float *aa [[buffer(0)]],device const float *ab [[buffer(1)]],device const float *dir [[buffer(2)]],device const float *bw [[buffer(3)]],device const float *five [[buffer(4)]],device const float *dm [[buffer(5)]],device float *out [[buffer(6)]],constant Args &a [[buffer(7)]],uint2 p [[thread_position_in_grid]]) {
 uint x=p.x,y=p.y,w=a.width,h=a.height;if(x>=w||y>=h)return;uint i=y*w+x;float g0=0;
 if(y>=4&&y<h-4&&x<w-8)g0=fma(dir[i],ab[i]-aa[i],aa[i]);float weight=max(bw[i],dm[i]);out[i]=fma(weight,five[i]-g0,g0);
}
kernel void chroma_inputs(device const float *s [[buffer(0)]],device float *hh [[buffer(1)]],device float *vv [[buffer(2)]],device float *ha [[buffer(3)]],device float *va [[buffer(4)]],constant Args &a [[buffer(5)]],uint2 p [[thread_position_in_grid]]) {
 int x=p.x,y=p.y,w=a.width,h=a.height;if(x>=w||y>=h)return;int i=y*w+x;float h1=0,v1=0,hav=0,vav=0;
 if(x<2||x>=w-2)h1=s[i];else h1=(fma(s[i],6.0f,s[i-2])+s[i+2])*0.125f;
 if(x>=2&&x<w-2&&y>=2&&y<h-2)v1=(fma(s[i],6.0f,s[i-2*w])+s[i+2*w])*0.125f;
 if(x>=2&&x<w-2)hav=(s[i-1]+s[i+1])*0.5f;
 if(y>=2&&y<h-2)vav=(s[i-w]+s[i+w])*0.5f;
 hh[i]=h1;vv[i]=v1;ha[i]=hav;va[i]=vav;
}
inline float soft(float a,float b,float c){float d=abs(a-c)+abs(b-c);return d==0.0f?0.5f:abs(b-c)/d;}
inline float bl(float base,float other,float a,float b,float weight){return fma(weight,other-(a+b),base);}
kernel void chroma_finish(device const float *hh [[buffer(0)]],device const float *vv [[buffer(1)]],device const float *ha [[buffer(2)]],device const float *va [[buffer(3)]],device const float *weight [[buffer(4)]],device const float *green [[buffer(5)]],device ushort *r [[buffer(6)]],device ushort *g [[buffer(7)]],device ushort *b [[buffer(8)]],constant Args &a [[buffer(9)]],uint2 p [[thread_position_in_grid]]) {
 int x=p.x,y=p.y,w=a.width,h=a.height;if(x>=w||y>=h)return;int i=y*w+x;float oa=0,ob=0,oc=0,od=0;
 bool vx=x>=2&&((x&1)?x<w-1:x<w-2),vy=y>=2&&((y&1)?y<h-1:y<h-2);
 if(vx&&vy){
#define H(dx,dy) get(hh,x+(dx),y+(dy),w,h)
#define V(dx,dy) get(vv,x+(dx),y+(dy),w,h)
#define A(dx,dy) get(ha,x+(dx),y+(dy),w,h)
#define B(dx,dy) get(va,x+(dx),y+(dy),w,h)
 if(!((x^y)&1)) {
 float hd=A(0,1)-H(0,1),wt=soft(H(0,-1),H(0,1),A(0,0));float vd=B(1,0)-V(1,0),wv=soft(V(-1,0),V(1,0),B(0,0));
 float hv=bl(hd,A(0,-1),H(0,-1),hd,wt),vert=bl(vd,B(-1,0),V(-1,0),vd,wv);
 if(!(y&1)){oa=H(0,0)-A(0,0);ob=V(0,0)-B(0,0);oc=hv;od=vert;}else{oa=hv;ob=vert;oc=H(0,0)-A(0,0);od=V(0,0)-B(0,0);}
 }else{
 float hd=H(0,1)-A(0,1),wt=soft(A(0,-1),A(0,1),H(0,0));float vd=V(1,0)-B(1,0),wv=soft(B(-1,0),B(1,0),V(0,0));
 float hv=bl(hd,H(0,-1),A(0,-1),hd,wt),vert=bl(vd,V(-1,0),B(-1,0),vd,wv);
 if(!(y&1)){oa=A(0,0)-H(0,0);ob=vert;oc=hv;od=B(0,0)-V(0,0);}else{oa=hv;ob=B(0,0)-V(0,0);oc=A(0,0)-H(0,0);od=vert;}
 }
#undef H
#undef V
#undef A
#undef B
 }
 float gv=green[i],rg=fma(weight[i],ob-oa,oa),bg=fma(weight[i],od-oc,oc);r[i]=ushort(clamp(rg+gv,0.0f,16383.0f));g[i]=ushort(clamp(gv,0.0f,16383.0f));b[i]=ushort(clamp(bg+gv,0.0f,16383.0f));
}
// Move the Bayer tile directly on the GPU, preserving the sensor phase at the image edge.
kernel void load_tile(device const ushort *image [[buffer(0)]],device ushort *tile [[buffer(1)]],constant Args &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
 if(p.x>=a.width||p.y>=a.height)return;
 int x=int(a.tile_x)+int(p.x)-int(a.halo), y=int(a.tile_y)+int(p.y)-int(a.halo);
 int w=int(a.image_width),h=int(a.image_height);
 x=x<0?(x&1):(x>=w?(w-1)-((w-1^x)&1):x);
 y=y<0?(y&1):(y>=h?(h-1)-((h-1^y)&1):y);
 tile[p.y*a.width+p.x]=image[y*w+x];
}
kernel void store_tile(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],device ushort *R [[buffer(3)]],device ushort *G [[buffer(4)]],device ushort *B [[buffer(5)]],constant Args &a [[buffer(6)]],uint2 p [[thread_position_in_grid]]) {
 if(p.x>=a.tile_width||p.y>=a.tile_height)return;
 uint src=(p.y+a.halo)*a.width+p.x+a.halo,dst=(p.y+a.tile_y)*a.image_width+p.x+a.tile_x;
 R[dst]=r[src];G[dst]=g[src];B[dst]=b[src];
}
