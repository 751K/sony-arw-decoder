// Sony near-saturation colour suppression, integer RAW path.
struct SscsArgs { uint width,height; int t,tr,tg,tb,wr,wg,wb; };
inline long sscs_div(long a,long b) { return b==0 ? 0 : a/b; }
inline void sscs_pixel(thread ushort &r,thread ushort &g,thread ushort &b,constant SscsArgs &a) {
    long r0=r,g0=g,b0=b;
    if(r0>a.tr && g0>a.tg && b0>a.tb) {
        long kg=min(sscs_div((g0-long(a.tg))*128,long(a.t)-a.tg),128L);
        bool red=r0<=b0;long c=red?r0:b0,tc=red?a.tr:a.tb;
        long krb=c>a.t?128L:sscs_div((c-tc)*128,long(a.t)-tc);
        long f=128-((krb*kg)>>7);
        long l=long(a.wr)*r0+long(a.wg)*g0+long(a.wb)*b0;
        long dr=f*(r0-g0),db=f*(b0-g0);
        long g2=((l*128)-dr*a.wr-db*a.wb)>>14;
        long r2=((g2*128)+dr)>>7,b2=((g2*128)+db)>>7;
        r0=r2;g0=g2;b0=b2;
    }
    r=ushort(ulong(r0)&0xfffful);
    g=ushort(ulong(g0)&0xfffful);
    b=ushort(ulong(b0)&0xfffful);
}
kernel void sscs_apply(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],device ushort *out [[buffer(3)]],constant SscsArgs &a [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    uint i=p.y*a.width+p.x;ushort r0=r[i],g0=g[i],b0=b[i];
    sscs_pixel(r0,g0,b0,a);
    out[3*i]=r0;out[3*i+1]=g0;out[3*i+2]=b0;
}
// GeoCorr sampling followed by SSCS in the same thread. The sampled values are
// stored in ushort before SSCS, matching the two-kernel path exactly.
struct GeoSscsArgs { GeoArgs geo; SscsArgs sscs; };
kernel void geo_sscs_apply(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],
                           device const float *tr [[buffer(3)]],device const float *tg [[buffer(4)]],device const float *tb [[buffer(5)]],
                           device ushort *out [[buffer(6)]],constant GeoSscsArgs &a [[buffer(7)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.geo.output_width||p.y>=a.geo.output_height)return;
    float px=float(a.geo.region_x)+float(p.x),py=float(a.geo.region_y)+float(p.y);
    float dx=px-a.geo.cx,dy=py-a.geo.cy;
    float radius=sqrt(fma(dy,dy,dx*dx)),t=(a.geo.s2*radius)/a.geo.step;
    int i0=int(floor(max(t,0.0f)));bool inside=i0<int(a.geo.n)-1;
    int i=clamp(i0,0,int(a.geo.n)-2);float frac=t-float(i);
    ushort r0=geo_channel(r,tr,a.geo,px,py,i,frac,inside,0);
    ushort g0=geo_channel(g,tg,a.geo,px,py,i,frac,inside,1);
    ushort b0=geo_channel(b,tb,a.geo,px,py,i,frac,inside,2);
    sscs_pixel(r0,g0,b0,a.sscs);
    uint o=p.y*a.geo.output_width+p.x;out[3*o]=r0;out[3*o+1]=g0;out[3*o+2]=b0;
}
