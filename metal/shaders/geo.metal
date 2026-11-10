// Sony GeoCorr: radial distortion and lateral chromatic aberration.
struct GeoArgs {
    uint width,height,output_width,output_height;
    int origin_x,origin_y,region_x,region_y;
    float cx,cy,s,s2,step;
    uint n;
    float off[6];
};
inline ushort geo_sample(device const ushort *plane, uint w, uint h, float fx, float fy) {
    int x0=clamp(int(fx),0,int(w)-1),y0=clamp(int(fy),0,int(h)-1);
    int x1=min(x0+1,int(w)-1),y1=min(y0+1,int(h)-1);
    float v00=float(plane[y0*w+x0]),v01=float(plane[y0*w+x1]);
    float v10=float(plane[y1*w+x0]),v11=float(plane[y1*w+x1]);
    float ax=fx-float(x0),bx=float(x1)-fx;
    float top=fma(bx,v00,ax*v01),bot=fma(ax,v11,bx*v10);
    float v=fma(bot,fy-float(y0),top*(float(y1)-fy));
    return ushort(uint(int(trunc(v))) & 0xffffu);
}
inline ushort geo_channel(device const ushort *plane,device const float *table,constant GeoArgs &a,
                           float px,float py,int i,float frac,bool inside,int k) {
    float f=inside?fma(table[i+1]-table[i],frac,table[i]):table[a.n-1];
    float sx=fma(f*((a.off[2*k]+px)-a.cx),a.s,a.cx);
    float sy=fma(f*((a.off[2*k+1]+py)-a.cy),a.s,a.cy);
    return geo_sample(plane,a.width,a.height,sx-float(a.origin_x),sy-float(a.origin_y));
}
kernel void geo_apply(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],
                      device const float *tr [[buffer(3)]],device const float *tg [[buffer(4)]],device const float *tb [[buffer(5)]],
                      device ushort *ro [[buffer(6)]],device ushort *go [[buffer(7)]],device ushort *bo [[buffer(8)]],
                      constant GeoArgs &a [[buffer(9)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.output_width||p.y>=a.output_height)return;
    float px=float(a.region_x)+float(p.x),py=float(a.region_y)+float(p.y);
    float dx=px-a.cx,dy=py-a.cy;
    float radius=sqrt(fma(dy,dy,dx*dx)),t=(a.s2*radius)/a.step;
    int i0=int(floor(max(t,0.0f)));bool inside=i0<int(a.n)-1;
    int i=clamp(i0,0,int(a.n)-2);float frac=t-float(i);uint o=p.y*a.output_width+p.x;
    ro[o]=geo_channel(r,tr,a,px,py,i,frac,inside,0);
    go[o]=geo_channel(g,tg,a,px,py,i,frac,inside,1);
    bo[o]=geo_channel(b,tb,a,px,py,i,frac,inside,2);
}
