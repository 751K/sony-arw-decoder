// Sony near-saturation colour suppression, integer RAW path.
struct SscsArgs { uint width,height; int t,tr,tg,tb,wr,wg,wb; };
inline long sscs_div(long a,long b) { return b==0 ? 0 : a/b; }
kernel void sscs_apply(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],device ushort *out [[buffer(3)]],constant SscsArgs &a [[buffer(4)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;
    uint i=p.y*a.width+p.x;long r0=r[i],g0=g[i],b0=b[i];
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
    out[3*i]=ushort(ulong(r0)&0xfffful);
    out[3*i+1]=ushort(ulong(g0)&0xfffful);
    out[3*i+2]=ushort(ulong(b0)&0xfffful);
}
