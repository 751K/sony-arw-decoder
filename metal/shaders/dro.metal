// DRO: raw analysis grid and local tone mapping.
struct DroArgs {
    uint width,height,raw_width,raw_height;
    int origin_x,origin_y;
    uint subpx_x,subpx_y,org_x,org_y,per_x,per_y;
    int black_r,black_g,black_b;
    float log2_max,strength,k_up,k_down,scale;
};

// Correctly rounded float32 division (the reference's numpy float32 '/').
// Metal's division may be off by an ulp; the candidate with the smallest
// exact fma remainder is the round-to-nearest quotient (ties to even).
inline float dro_div(float a,float b) {
    float q=a/b;if(!isfinite(q)||q==0.0f)return q;
    uint sign=as_type<uint>(q)&0x80000000u,mag=as_type<uint>(q)&0x7fffffffu;
    float best=q,br=abs(fma(-q,b,a));
    for(int k=-2;k<=2;k++){                    // neighbouring floats of the same sign
        if(k==0||(k<0&&mag<uint(-k)))continue;
        float c=as_type<float>(sign|(mag+uint(k)));
        float r=abs(fma(-c,b,a));
        if(r<br||(r==br&&(as_type<uint>(c)&1u)==0u)){best=c;br=r;}
    }
    return best;
}
// float32(exp2(double(e))) as numpy computes it, without double: 2^frac(e)
// in Q62 fixed point (256-entry table times a degree-6 series, error ~2^-60),
// rounded to 53 bits and then to 24 bits, both to nearest even.
constant ulong dro_exp2_tab[256]={0x4000000000000000UL,0x402c6be96af2fb58UL,0x4058f6a7ecccd5b6UL,0x4085a050ec0a036aUL,0x40b268f9de0183baUL,0x40df50b846ef302bUL,0x410c57a1b9fe12f6UL,0x41397dcbd952c4a6UL,0x4166c34c5615d0ecUL,0x41942838f07e22a3UL,0x41c1aca777db771bUL,0x41ef50adcaa0d899UL,0x421d1461d66f2023UL,0x424af7d9981f7e98UL,0x4278fb2b1bce0d15UL,0x42a71e6c7ce464afUL,0x42d561b3e6243d8aUL,0x4303c51791b21550UL,0x433248adc91fdd02UL,0x4360ec8ce577ae44UL,0x438fb0cb4f468808UL,0x43be957f7ea712b6UL,0x43ed9abffb4c6bc9UL,0x441cc0a35c8cf8eaUL,0x444c0740496d4294UL,0x447b6ead78aad636UL,0x44aaf701b0c72feeUL,0x44daa053c812abd2UL,0x450a6abaa4b77ecdUL,0x453a564d3cc4b723UL,0x456a632296394492UL,0x459a9151c70f0819UL,0x45cae0f1f545eb73UL,0x45fb521a56ef0042UL,0x462be4e23237a6efUL,0x465c9960dd74bd4bUL,0x468d6fadbf2dd4f3UL,0x46be67e04e28717cUL,0x46ef821011734e6bUL,0x4720be54a071ad01UL,0x47521cc5a2e6a9e0UL,0x47839d7ad1009a90UL,0x47b5408bf36472e2UL,0x47e70610e3393241UL,0x4818ee218a3358eeUL,0x484af8d5e2a06536UL,0x487d2645f7725895UL,0x48af7689e44b44eaUL,0x48e1e9b9d588e19bUL,0x49147fee085028d6UL,0x4947393eca98fcd6UL,0x497a15c47b39d53eUL,0x49ad159789f37496UL,0x49e038d0777ca5e0UL,0x4a137f87d58e025bUL,0x4a46e9d646edbf6aUL,0x4a7a77d47f7b84b1UL,0x4aae299b443c4a6bUL,0x4ae1ff436b663ff7UL,0x4b15f8e5dc6cbab0UL,0x4b4a169b900c2d00UL,0x4b7e587d905625d1UL,0x4bb2bea4f8bd5847UL,0x4be7492af621abd6UL,0x4c1bf828c6dc54b8UL,0x4c50cbb7bacbf4cbUL,0x4c85c3f13360c4d5UL,0x4cbae0eea3a8c63cUL,0x4cf022c9905bfd32UL,0x4d25899b8fe8b365UL,0x4d5b157e4a7fc325UL,0x4d90c68b7a20eb1eUL,0x4dc69cdceaa72a9cUL,0x4dfc988c79d52661UL,0x4e32b9b417619617UL,0x4e69006dc503ba65UL,0x4e9f6cd3967fdba8UL,0x4ed5feffb1b3d155UL,0x4f0cb70c4ea39210UL,0x4f439513b785cc80UL,0x4f7a993048d088d7UL,0x4fb1c37c7145d328UL,0x4fe91412b2006e83UL,0x50208b0d9e8090dfUL,0x50582887dcb8a7e1UL,0x508fec9c251a267cUL,0x50c7d76542a25b72UL,0x50ffe8fe12e750c4UL,0x513821818624b40cUL,0x5170810a9f48c7cfUL,0x51a907b474015dc9UL,0x51e1b59a2cc8da3eUL,0x521a8ad704f3404fUL,0x525387864abb4762UL,0x528cabc35f4f799dUL,0x52c5f7a9b6df5b77UL,0x52ff6b54d8a89c75UL,0x533906e05f045106UL,0x5372ca67f774358fUL,0x53acb60762affaaeUL,0x53e6c9da74b29ab5UL,0x542105fd14c7b869UL,0x545b6a8b3d990704UL,0x5495f7a0fd3bbb8eUL,0x54d0ad5a753e077cUL,0x550b8bd3dab49cbdUL,0x5546932976483b15UL,0x5581c377a44346f0UL,0x55bd1cdad49f699cUL,0x55f89f6f8b133af4UL,0x56344b525f1ff495UL,0x5670209ffc1f2e82UL,0x56ac1f752150a563UL,0x56e847eea1e80a49UL,0x57249a29651adc07UL,0x57611642662e4a33UL,0x579dbc56b48521baUL,0x57da8c8373adc331UL,0x581786e5db7022c2UL,0x5854ab9b37dbd1e5UL,0x5891fac0e95612c8UL,0x58cf747464a7f584UL,0x590d18d3330c7f1eUL,0x594ae7faf23eda52UL,0x5988e20954889245UL,0x59c7071c20cfd70cUL,0x5a05575132a5cc20UL,0x5a43d2c67a54e0c0UL,0x5a827999fcef3242UL,0x5ac14be9d45cf869UL,0x5b0049d42f6afbb6UL,0x5b3f737751d915c8UL,0x5b7ec8f19468bbc9UL,0x5bbe4a6164eb92f4UL,0x5bfdf7e546520f3eUL,0x5c3dd19bd0ba1c1aUL,0x5c7dd7a3b17dcf75UL,0x5cbe0a1bab4226dfUL,0x5cfe69229605cef5UL,0x5d3ef4d75f2ff504UL,0x5d7fad59099f22feUL,0x5dc092c6adb825b8UL,0x5e01a53f7974fd86UL,0x5e42e4e2b073d935UL,0x5e8451cfac061b5fUL,0x5ec5ec25db3f6a41UL,0x5f07b404c304c9f1UL,0x5f49a98bfe1bc11cUL,0x5f8bccdb3d398841UL,0x5fce1e124712437cUL,0x60109d50f86846d8UL,0x60534ab7441b6546UL,0x6096266533384a2bUL,0x60d9307ae507dd9aUL,0x611c69188f1eb339UL,0x615fd05e7d6c83e4UL,0x61a3666d124bb204UL,0x61e72b64c690d8afUL,0x622b1f66299a6599UL,0x626f4291e1603dd5UL,0x62b39508aa836d6fUL,0x62f816eb585de1edUL,0x633cc85ad5122fbdUL,0x6381a978219b628fUL,0x63c6ba6455dcd8aeUL,0x640bfb40a0b22959UL,0x64516c2e47ff1623UL,0x64970d4ea8bf8766UL,0x64dcdec3371793d1UL,0x6522e0ad7e639312UL,0x6569132f21483ba7UL,0x65af7669d9c2cbe4UL,0x65f60a7f79393e2eUL,0x663ccf91e88a8873UL,0x6683c5c3281ee6e9UL,0x66caed354ff83222UL,0x6712460a8fc24072UL,0x6759d0652ee352afUL,0x67a18c678c8c8c61UL,0x67e97a341fca775bUL,0x683199ed779592caUL,0x6879ebb63ae2edc6UL,0x68c26fb128b4cd63UL,0x690b2601182b5e56UL,0x69540ec8f895722dUL,0x699d2a2bd181482aUL,0x69e6784cc2cd61bdUL,0x6a2ff94f04b962b0UL,0x6a79ad55e7f6fd10UL,0x6ac39484d5bae8c5UL,0x6b0daeff4fcde703UL,0x6b57fce8f09dd17fUL,0x6ba27e656b4eb57aUL,0x6bed33988bcbfab5UL,0x6c381ca636d99642UL,0x6c8339b26a25494eUL,0x6cce8ae13c57ebdbUL,0x6d1a1056dd26c382UL,0x6d65ca379564e639UL,0x6db1b8a7c714a92bUL,0x6dfddbcbed791babUL,0x6e4a33c89d278e49UL,0x6e96c0c284192610UL,0x6ee382de69bc7bf8UL,0x6f307a412f074892UL,0x6f7da70fce881bf9UL,0x6fcb096f5c782210UL,0x7018a18506ccf314UL,0x70666f76154a7089UL,0x70b47367e994ae92UL,0x7102ad7fff41e9b4UL,0x71511de3ebec8913UL,0x719fc4b95f452d29UL,0x71eea2262324cb0cUL,0x723db6501b9ed447UL,0x728d015d47136b40UL,0x72dc8373be41a454UL,0x732c3cb9b459d38bUL,0x737c2d55770fe711UL,0x73cc556d6eadce5fUL,0x741cb5281e25ee34UL,0x746d4cac2325a155UL,0x74be1c203627c62bUL,0x750f23ab2a87593eUL,0x75606373ee921c97UL,0x75b1dba18b9b4c20UL,0x76038c5b260e5eeeUL,0x765575c7fd81d5a0UL,0x76a7980f6cca15c2UL,0x76f9f358ea0c5249UL,0x774c87cc06d1812eUL,0x779f559070195e36UL,0x77f25ccdee6d7ae6UL,0x78459dac65f45bb5UL,0x78991853d684a285UL,0x78ecccec5bb84661UL,0x7940bb9e2cffd89dUL,0x7994e4919db5d74eUL,0x79e947ef1d320d2dUL,0x7a3de5df36dcfeefUL,0x7a92be8a92436616UL,0x7ae7d219f329b948UL,0x7b3d20b6399fc237UL,0x7b92aa886214411cUL,0x7be86fb985689ddcUL,0x7c3e7072d904a6ceUL,0x7c94acddaeea5d3aUL,0x7ceb252375c9cf9bUL,0x7d41d96db915019dUL,0x7d98c9e62113e1faUL,0x7deff6b672f84e24UL,0x7e47600890f223deUL,0x7e9f06067a4360baUL,0x7ef6e8da4b544f9aUL,0x7f4f08ae3dc7c426UL,0x7fa765aca88f6453UL};
constant ulong dro_ln2_q64=0xb17217f7d1cf79acUL;

inline ulong dro_mulq(ulong a,ulong b) { return (mulhi(a,b)<<2)|((a*b)>>62); }
inline float dro_exp2(float e) {
    if(isnan(e))return e;
    if(e>=128.0f)return INFINITY;
    if(e<-149.0f)return 0.0f;
    float fl=floor(e);int n=int(fl);
    // frac(e) in Q62, from e's bits: |e| = m * 2^sh / 2^62; the low 62 bits are the fraction
    // of |e| (bits below 2^-62 dropped); for e < 0 the fraction is one minus that.
    uint bits=as_type<uint>(e),ex=(bits>>23)&0xffu;ulong em=ex?((bits&0x7fffffu)|0x800000u):0;
    int sh=int(ex)-150+62;
    ulong Fm=sh>=64?0:(sh>=0?((em<<sh)&((1UL<<62)-1)):(-sh>=64?0:em>>(-sh)));
    if(e<0.0f&&Fm==0&&fl!=e)Fm=1;                      // |e| below 2^-62: frac(e) = 1 - tiny
    ulong F=e<0.0f?(((1UL<<62)-Fm)&((1UL<<62)-1)):Fm;
    uint j=uint(F>>54);ulong t=mulhi(F&((1UL<<54)-1),dro_ln2_q64);  // t = r * ln2, Q62
    const ulong one=1UL<<62;ulong q=one;
    for(ulong k=6;k>=1;k--)q=one+dro_mulq(t,q)/k;
    ulong z=dro_mulq(dro_exp2_tab[j],q);                 // 2^f in Q62, [2^62, 2^63)
    if(z>=(1UL<<63)){z>>=1;n++;}
    ulong m=z>>10,rem=z&1023;                            // to 53 bits
    if(rem>512||(rem==512&&(m&1)))m++;
    if(m==(1UL<<53)){m>>=1;n++;}
    ulong m24=m>>29,r24=m&((1UL<<29)-1);                 // to 24 bits
    if(r24>(1UL<<28)||(r24==(1UL<<28)&&(m24&1)))m24++;
    if(m24==(1UL<<24)){m24>>=1;n++;}
    return ldexp(float(m24),n-23);
}
kernel void dro_math_test(device const float *a [[buffer(0)]],device const float *b [[buffer(1)]],device float *e2 [[buffer(2)]],device float *q [[buffer(3)]],constant uint &n [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=n)return;e2[i]=dro_exp2(a[i]);q[i]=dro_div(a[i],b[i]);
}
inline float dro_log(float v,device const float *tab) {
    if(v<0.0f)return 0.0f;
    if(v>=16384.0f)return log2(v);
    float t=v*0.125f;uint i=uint(t);float f=t-float(i);
    return fma(f,tab[i+1]-tab[i],tab[i]);
}
kernel void dro_subcells(device const ushort *raw [[buffer(0)]], device const float *logtab [[buffer(1)]], device float *sub [[buffer(2)]],constant DroArgs &a [[buffer(3)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=112||p.y>=84)return;
    float inv=dro_div(1.0f,float(a.per_x*a.per_y)), acc=0.0f;
    for(uint y=0;y<a.per_y;y++)for(uint x=0;x<a.per_x;x++){
        uint xx=a.org_x+4*(p.x*a.per_x+x),yy=a.org_y+4*(p.y*a.per_y+y),i=yy*a.raw_width+xx;
        float r=float(max(int(raw[i])-a.black_r,0));
        float g=float(max(((int(raw[i+1])+int(raw[i+a.raw_width]))>>1)-a.black_g,0));
        float b=float(max(int(raw[i+a.raw_width+1])-a.black_b,0));
        float v=fma(b+g,0.057f,r*0.1495f);
        acc=fma(dro_log(v,logtab),inv,acc);
    }
    sub[p.y*112+p.x]=acc;
}
kernel void dro_histogram(device const float *sub [[buffer(0)]],device float *hist [[buffer(1)]],constant DroArgs &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=8||p.y>=6)return;
    float bins[14]={0};float inc=dro_div(1.0f,196.0f),inv=dro_div(1.0f,a.log2_max);
    for(uint y=0;y<14;y++)for(uint x=0;x<14;x++){
        float v=sub[(p.y*14+y)*112+p.x*14+x];int b=int(max((v*14.0f)*inv,0.0f));if(b==14)b--;
        bins[clamp(b,0,13)]+=inc;
    }
    for(uint j=0;j<14;j++)hist[(p.y*8+p.x)*14+j]=bins[j];
}
kernel void dro_smooth(device const float *hist [[buffer(0)]],device float *smooth [[buffer(1)]],constant DroArgs &a [[buffer(2)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=8||p.y>=6)return;
    int x=p.x,y=p.y;for(int j=0;j<14;j++){
#define H(dx,dy) hist[((y+(dy))*8+x+(dx))*14+j]
        float c=H(0,0),v;
        if(y==0||y==5){float vert=y==0?H(0,1):H(0,-1);
            if(x==0)v=(fma(c,2.0f,H(1,0))+vert)*0.25f;
            else if(x==7)v=(fma(c,2.0f,H(-1,0))+vert)*0.25f;
            else v=(fma(c,3.0f,H(-1,0))+(H(1,0)+vert))*0.16666667f;
        }else if(x==0)v=(fma(c,3.0f,H(1,0))+(H(0,-1)+H(0,1)))*0.16666667f;
        else if(x==7)v=(fma(c,3.0f,H(-1,0))+(H(0,-1)+H(0,1)))*0.16666667f;
        else v=(fma(c,4.0f,H(-1,0))+((H(1,0)+H(0,-1))+H(0,1)))*0.125f;
#undef H
        smooth[(y*8+x)*14+j]=v;
    }
}
kernel void dro_weights(device const float *sm [[buffer(0)]],device float *W [[buffer(1)]],device float *S [[buffer(2)]],constant DroArgs &a [[buffer(3)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=8||p.y>=6)return;
    constexpr float weight[9]={0.8828129768371582f,0.6054689884185791f,0.3242189884185791f,0.1367190033197403f,0.04296899959445f,0.011718999594449997f,0.0039059999398887157f,0.0039059999398887157f,0.0039059999398887157f};
    constexpr float blend[14]={1,.75,.5,.25,0,0,0,0,0,.25,.5,.75,1,1};
    int pos=(p.y*8+p.x)*14;
    for(int j=0;j<14;j++){float w=0,s=0;for(int k=0;k<14;k++){float hk=sm[pos+k],wt=weight[min(abs(j-k),8)];w=fma(hk,wt,w);s=fma(hk*wt,float(k),s);}S[pos+j]=fma(blend[j],fma(w,float(j),-s),s);W[pos+j]=w;}
}
inline int4 dro_axis(int p,int c,int cw,int n){
    int i0=p<c?0:(p<c+(n-1)*cw?(p-c)/max(cw,1):n-1);
    int i1=p<c?0:(p<c+(n-1)*cw?i0+1:n-1);
    return int4(i0,i1,abs(p-(c+i0*cw)),abs(p-(c+i1*cw)));
}
inline float dro_cell(device const float *table,int y,int x,int bin){return bin>=14?0.0f:table[(y*8+x)*14+bin];}
inline float dro_interp(device const float *table,int4 ax,int4 ay,int bin){
    float a0=dro_cell(table,ay.x,ax.x,bin),a1=dro_cell(table,ay.x,ax.y,bin);
    float a2=dro_cell(table,ay.y,ax.x,bin),a3=dro_cell(table,ay.y,ax.y,bin);
    int sx=ax.z+ax.w,sy=ay.z+ay.w;float inv=sx?dro_div(1.0f,float(sx)):0;
    float top=sx?fma(a1,float(ax.z),a0*float(ax.w))*inv:a0;
    float bot=sx?fma(a3,float(ax.z),a2*float(ax.w))*inv:a2;
    return sy?dro_div(fma(bot,float(ay.z),top*float(ay.w)),float(sy)):top;
}
kernel void dro_apply(device const ushort *r [[buffer(0)]],device const ushort *g [[buffer(1)]],device const ushort *b [[buffer(2)]],device const float *weights [[buffer(3)]],device const float *sums [[buffer(4)]],device const float *curve [[buffer(5)]],device const float *logtab [[buffer(6)]],device ushort *ro [[buffer(7)]],device ushort *go [[buffer(8)]],device ushort *bo [[buffer(9)]],constant DroArgs &a [[buffer(10)]],uint2 p [[thread_position_in_grid]]) {
    if(p.x>=a.width||p.y>=a.height)return;uint i=p.y*a.width+p.x;
    float R=float(r[i]>>1),G=float(g[i]>>1),B=float(b[i]>>1);
    float Y=fma(0.057f,B,fma(0.2935f,G,0.1495f*R));Y=Y+Y;
    float L=dro_log(Y,logtab);int px=int(a.scale*float(int(p.x)+a.origin_x)),py=int(a.scale*float(int(p.y)+a.origin_y));
    int cw=a.subpx_x*14,ch=a.subpx_y*14;int4 ax=dro_axis(px,a.org_x+(cw>>1),cw,8),ay=dro_axis(py,a.org_y+(ch>>1),ch,6);
    float fl=floor(L),f=L-fl;int bin=int(max(fl,0.0f)),bin1=bin<12?bin+1:13;
    float w0=dro_interp(weights,ax,ay,bin),w1=dro_interp(weights,ax,ay,bin1);
    float s0=dro_interp(sums,ax,ay,bin),s1=dro_interp(sums,ax,ay,bin1);
    float W=fma(w1,f,fma(-f,w0,w0)),S=fma(s1,f,s0*(1.0f-f));
    float local=W==0.0f?a.log2_max:dro_div(S,W);int idx=int(max(local*8.0f,0.0f)),ic=min(idx,102);
    float t=fma(float(ic),-0.125f,local)*8.0f,cv=idx<103?fma(t,curve[ic+1]-curve[ic],curve[ic]):curve[103];
    float e=fma(L-local,a.strength,-L)+cv,gain=dro_exp2(e),k=(gain-1.0f)>0?a.k_up:a.k_down,m=fma(gain,k,-k);
    ro[i]=ushort(uint(clamp(fma(m,R,R),0.0f,16383.0f))<<1);
    go[i]=ushort(uint(clamp(fma(m,G,G),0.0f,16383.0f))<<1);
    bo[i]=ushort(uint(clamp(fma(m,B,B),0.0f,16383.0f))<<1);
}
