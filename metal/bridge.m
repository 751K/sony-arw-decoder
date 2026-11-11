#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { uint32_t width,height; int32_t black[4]; float gains[4]; int32_t scale; float blend_k,blend_lo,blend_hi,blend_half; float k1,k2,k3,k4,k5,k6; uint32_t image_width,image_height,tile_x,tile_y,tile_width,tile_height,halo,src_stride,src_origin; } Args;
typedef struct { uint32_t width,height; int32_t off[3],gain[3],lim[3],wm[3]; float s; uint32_t raw_stride,raw_origin; } NrArgs;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLCommandQueue> aux_queue;
static id<MTLSharedEvent> grid_event;
static id<MTLCommandBuffer> batch_cmd;
static NSMutableArray *flushed_cmds;      // batch buffers committed early, not yet waited on
// RAW streaming: raw_event counts decoded sensor rows (stream base + rows).
static id<MTLSharedEvent> raw_event;
static uint64_t raw_stream_base;
static uint64_t grid_event_value;
static uint64_t pending_grid_value;
#define NPIPE 58
static id<MTLComputePipelineState> pipelines[NPIPE];
static char error_text[2048];
const char *sonylook_metal_error(void) { return error_text; }
static int fail(NSString *s) { snprintf(error_text,sizeof(error_text),"%s",s.UTF8String); return -1; }
// ---------------------------------------------------------------------------
// GPU profiling (SONYLOOK_GPU_PROFILE=1): every encoder records GPU timestamps
// at its start and end; sonylook_metal_profile_report() sums them per kernel
// (label = pipeline index; 100/101 = a whole ITP tile encoder). Only valid
// once the profiled work has completed. Off by default: no sampling at all.
#define PROF_CAP 4096          // the sample buffer holds at most 32 KB
static int prof_on=-1;static id<MTLCounterSampleBuffer> prof_sb;static uint32_t prof_n;static int prof_label[PROF_CAP/2];
static double prof_ticks[128];static uint64_t prof_calls[128];static MTLTimestamp prof_c0,prof_g0;static NSArray *pipeline_names;
static int prof_init(void){
    if(prof_on>=0)return prof_on;
    prof_on=0;
    if(!getenv("SONYLOOK_GPU_PROFILE")||![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])return 0;
    id<MTLCounterSet> ts=nil;for(id<MTLCounterSet> c in device.counterSets)if([c.name isEqualToString:MTLCommonCounterSetTimestamp])ts=c;
    if(!ts)return 0;
    MTLCounterSampleBufferDescriptor *d=[MTLCounterSampleBufferDescriptor new];d.counterSet=ts;d.sampleCount=PROF_CAP;d.storageMode=MTLStorageModeShared;
    prof_sb=[device newCounterSampleBufferWithDescriptor:d error:nil];if(!prof_sb)return 0;
    [device sampleTimestamps:&prof_c0 gpuTimestamp:&prof_g0];
    prof_on=1;return 1;
}
static id<MTLComputeCommandEncoder> new_encoder(id<MTLCommandBuffer> cmd,int label){
    if(!prof_init()||prof_n+2>PROF_CAP)return [cmd computeCommandEncoder];
    MTLComputePassDescriptor *d=[MTLComputePassDescriptor computePassDescriptor];
    d.sampleBufferAttachments[0].sampleBuffer=prof_sb;
    d.sampleBufferAttachments[0].startOfEncoderSampleIndex=prof_n;d.sampleBufferAttachments[0].endOfEncoderSampleIndex=prof_n+1;
    prof_label[prof_n/2]=label;prof_n+=2;
    return [cmd computeCommandEncoderWithDescriptor:d];
}
// Fold completed samples into the per-kernel totals (call after the GPU work finished).
void sonylook_metal_profile_collect(void){
    if(prof_on!=1||!prof_n)return;
    NSData *r=[prof_sb resolveCounterRange:NSMakeRange(0,prof_n)];
    const MTLCounterResultTimestamp *t=(const MTLCounterResultTimestamp *)r.bytes;
    for(uint32_t k=0;k+1<prof_n;k+=2){int l=prof_label[k/2];if(l<0||l>=128)continue;
        if(t[k].timestamp==MTLCounterErrorValue||t[k+1].timestamp==MTLCounterErrorValue||t[k+1].timestamp<t[k].timestamp)continue;
        prof_ticks[l]+=(double)(t[k+1].timestamp-t[k].timestamp);prof_calls[l]++;}
    prof_n=0;
}
// Print per-kernel GPU milliseconds divided by `runs`, then reset.
void sonylook_metal_profile_report(uint32_t runs){
    sonylook_metal_profile_collect();
    if(prof_on!=1){fprintf(stderr,"GPU profiling unavailable (set SONYLOOK_GPU_PROFILE=1)\n");return;}
    MTLTimestamp c1,g1;[device sampleTimestamps:&c1 gpuTimestamp:&g1];
    // sampleTimestamps' CPU value is in nanoseconds; GPU ticks are scaled to it.
    double ns_per_tick=g1>prof_g0?(double)(c1-prof_c0)/(double)(g1-prof_g0):1.0;
    double total=0;for(int l=0;l<128;l++)total+=prof_ticks[l];
    fprintf(stderr,"GPU time per run (%u runs), total %.2f ms\n",runs,total*ns_per_tick/1e6/(runs?runs:1));
    for(int l=0;l<128;l++)if(prof_calls[l]){
        NSString *n=l==100?@"itp tile (9 kernels)":l==101?@"itp tile (3 passes)":(l<(int)pipeline_names.count?pipeline_names[l]:@"?");
        fprintf(stderr,"  %-24s %8.2f ms  %6llu encoders\n",n.UTF8String,prof_ticks[l]*ns_per_tick/1e6/(runs?runs:1),(unsigned long long)(prof_calls[l]/(runs?runs:1)));
        prof_ticks[l]=0;prof_calls[l]=0;}
}

static void encode_grid(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,const void *args,size_t len,uint32_t w,uint32_t h);
static void encode_band(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,const void *args,size_t len,uint32_t w,uint32_t row0,uint32_t rows);
// The ITP input streamed from a larger sensor raw that is still being decoded:
// image row y is source row y + src_y, available once raw_event >= base + that + 1.
typedef struct { const uint16_t *src; uint32_t src_stride,src_rows,src_x,src_y; uint64_t base; } RawStream;
static NSArray *new_buffers(size_t count,size_t length,MTLResourceOptions options);

// ---------------------------------------------------------------------------
// Buffer management.
//
// Resident arrays: sonylook_metal_alloc() hands Python the contents of a
// shared MTLBuffer, so NumPy arrays can live in GPU-visible memory. Inputs
// that point into a resident buffer are bound in place (in_buf) and outputs
// are written straight into it (out_buf); other pointers are staged through
// pooled buffers. Temporaries come from a pool and are returned to it by
// done(), so repeated calls do not allocate (and page-fault) new memory.
@interface SLBuf : NSObject
@property(nonatomic,strong) id<MTLBuffer> buf;
@property(nonatomic) NSUInteger off;
@end
@implementation SLBuf
- (void *)contents { return (uint8_t *)self.buf.contents+self.off; }
@end
static NSMutableArray *pool_bufs;       // idle buffers
static NSMutableArray *call_bufs;       // buffers lent out during the current call
static NSMutableDictionary *resident;   // contents address -> buffer
static size_t pool_bytes;
static const size_t POOL_CAP=(size_t)3<<30;
static NSObject *buf_lock;
static size_t round_len(size_t len) { return ((len?len:1)+16383)&~(size_t)16383; }
static id<MTLBuffer> pool_take(size_t len,MTLResourceOptions mode) {
    len=round_len(len);MTLStorageMode sm=(mode&MTLResourceStorageModeMask)>>MTLResourceStorageModeShift;
    @synchronized(buf_lock) {
        if(!pool_bufs){pool_bufs=[NSMutableArray array];call_bufs=[NSMutableArray array];resident=[NSMutableDictionary dictionary];}
        NSUInteger best=NSNotFound;size_t bestlen=SIZE_MAX;
        for(NSUInteger i=0;i<pool_bufs.count;i++){id<MTLBuffer> b=pool_bufs[i];
            if(b.storageMode==sm&&b.length>=len&&b.length<=2*len&&b.length<bestlen){best=i;bestlen=b.length;}}
        if(best!=NSNotFound){id<MTLBuffer> b=pool_bufs[best];[pool_bufs removeObjectAtIndex:best];pool_bytes-=b.length;return b;}
    }
    return [device newBufferWithLength:len options:mode];
}
static void pool_put(id<MTLBuffer> b) {
    @synchronized(buf_lock) {
        [pool_bufs addObject:b];pool_bytes+=b.length;
        while(pool_bytes>POOL_CAP&&pool_bufs.count){id<MTLBuffer> o=pool_bufs[0];pool_bytes-=o.length;[pool_bufs removeObjectAtIndex:0];}
    }
}
// A temporary for this call.
static id<MTLBuffer> acquire(size_t len,MTLResourceOptions mode) {
    id<MTLBuffer> b=pool_take(len,mode);if(b){@synchronized(buf_lock){[call_bufs addObject:b];}}
    return b;
}
// Return this call's temporaries to the pool (the command buffer has completed).
static int done(int rc) {
    if(batch_cmd)return rc;
    @synchronized(buf_lock){for(id<MTLBuffer> b in call_bufs)pool_put(b);[call_bufs removeAllObjects];}
    return rc;
}
static id<MTLCommandBuffer> command_buffer(void) {
    return batch_cmd ? batch_cmd : [queue commandBuffer];
}
// For calls that must commit and read back immediately: refuse to run while a
// batch is open, since their buffer would execute ahead of the batched work.
static id<MTLCommandBuffer> own_command_buffer(void) {
    if(batch_cmd){fail(@"This Metal call cannot run inside a command batch; call sonylook_metal_end() first");return nil;}
    return [queue commandBuffer];
}
static int submit_command(id<MTLCommandBuffer> cmd) {
    if(batch_cmd)return 0;
    [cmd commit];[cmd waitUntilCompleted];
    return cmd.error ? fail(cmd.error.description) : 0;
}
int sonylook_metal_begin(void) {
    if(!device)return fail(@"Metal device unavailable");
    if(batch_cmd)return fail(@"Metal command batch already active");
    batch_cmd=[queue commandBuffer];
    return batch_cmd ? 0 : fail(@"Cannot create Metal command buffer");
}
// Commit the batch so far without waiting and keep batching into a new
// buffer on the same queue (which preserves order): the GPU can start on
// early stages while the host is still encoding or producing input.
int sonylook_metal_flush(void) {
    if(!batch_cmd)return fail(@"No Metal command batch is active");
    if(!flushed_cmds)flushed_cmds=[NSMutableArray array];
    [batch_cmd commit];[flushed_cmds addObject:batch_cmd];
    batch_cmd=[queue commandBuffer];
    return batch_cmd ? 0 : fail(@"Cannot create Metal command buffer");
}
static int finish_flushed(void) {
    int rc=0;
    for(id<MTLCommandBuffer> c in flushed_cmds){[c waitUntilCompleted];if(c.error&&!rc)rc=fail(c.error.description);}
    [flushed_cmds removeAllObjects];
    return rc;
}
int sonylook_metal_end(void) {
    if(!batch_cmd)return fail(@"No Metal command batch is active");
    id<MTLCommandBuffer> cmd=batch_cmd;
    batch_cmd=nil;
    int rc=submit_command(cmd);
    int frc=finish_flushed();
    done(0);
    return rc?rc:frc;
}
int sonylook_metal_abort(void) {
    batch_cmd=nil;
    finish_flushed();          // their temporaries must not be reused before they finish
    done(0);
    return 0;
}
// RAW streaming. begin returns the base value for a new stream; the decoder
// reports rows through raw_progress(ctx = base); release unblocks all waits
// (decode failure).
uint64_t sonylook_metal_raw_stream_begin(void) {
    raw_stream_base+=(uint64_t)1<<24;
    return raw_stream_base;
}
void sonylook_metal_raw_progress(void *ctx,uint32_t rows) {
    raw_event.signaledValue=(uint64_t)(uintptr_t)ctx+rows;
}
void sonylook_metal_raw_release(uint64_t base) {
    raw_event.signaledValue=base+((uint64_t)1<<24)-1;
}
int sonylook_metal_is_active(void) { return batch_cmd != nil; }
static SLBuf *find_resident(const void *p,size_t len) {
    @synchronized(buf_lock) {
        for(NSNumber *k in resident){id<MTLBuffer> b=resident[k];uintptr_t s=(uintptr_t)b.contents,q=(uintptr_t)p;
            if(q>=s&&q+len<=s+b.length&&((q-s)&15)==0){SLBuf *r=[SLBuf new];r.buf=b;r.off=q-s;return r;}}
    }
    return nil;
}
// Bind host data: in place when resident, otherwise copied into a pooled buffer.
static id in_buf(const void *p,size_t len) {
    SLBuf *r=find_resident(p,len);if(r)return r;
    id<MTLBuffer> b=acquire(len,MTLResourceStorageModeShared);if(b)memcpy(b.contents,p,len);
    return b;
}
// Output destination: the resident array itself, or a pooled buffer to copy back.
static id out_buf(void *p,size_t len) {
    SLBuf *r=find_resident(p,len);if(r)return r;
    return acquire(len,MTLResourceStorageModeShared);
}
static void copy_back(id o,void *p,size_t len) {
    if(batch_cmd)return;
    if(![o isKindOfClass:[SLBuf class]])memcpy(p,[o contents],len);
}
static void bind(id<MTLComputeCommandEncoder> e,id o,NSUInteger i) {
    if([o isKindOfClass:[SLBuf class]])[e setBuffer:((SLBuf *)o).buf offset:((SLBuf *)o).off atIndex:i];
    else [e setBuffer:o offset:0 atIndex:i];
}
void *sonylook_metal_alloc(size_t len) {
    if(!device)return NULL;
    id<MTLBuffer> b=pool_take(len,MTLResourceStorageModeShared);if(!b)return NULL;
    @synchronized(buf_lock){resident[@((uintptr_t)b.contents)]=b;}
    return b.contents;
}
void sonylook_metal_free(void *p) {
    id<MTLBuffer> b=nil;
    @synchronized(buf_lock){NSNumber *k=@((uintptr_t)p);b=resident[k];[resident removeObjectForKey:k];}
    if(b)pool_put(b);
}
// Drop all idle pooled buffers.
void sonylook_metal_trim(void) { @synchronized(buf_lock){[pool_bufs removeAllObjects];pool_bytes=0;} }
size_t sonylook_metal_pool_bytes(void) { @synchronized(buf_lock){return pool_bytes;} }
int sonylook_metal_init(const char *path) { @autoreleasepool {
    if (device) return 0;
    buf_lock=[NSObject new];
    id<MTLDevice> d=MTLCreateSystemDefaultDevice(); if(!d) return fail(@"Metal device unavailable");
    // path: the shaders directory. Prefer the precompiled metallib, then fall
    // back to concatenating the individual .metal sources.
    NSError *err=nil; NSMutableString *source=[NSMutableString string];
    NSString *dir=[NSString stringWithUTF8String:path];
    NSString *compiled=[dir stringByAppendingPathComponent:@"shaders.metallib"];
    id<MTLLibrary> library=nil;
    if([[NSFileManager defaultManager] fileExistsAtPath:compiled]) {
        library=[d newLibraryWithURL:[NSURL fileURLWithPath:compiled] error:&err];
        if(!library)return fail(err.description);
    }
    if(!library) {
        for(NSString *name in @[@"itp",@"dro",@"geo",@"sscs",@"look",@"spatial",@"marble",@"post",@"nr"]){
            NSString *src=[NSString stringWithContentsOfFile:[dir stringByAppendingPathComponent:[name stringByAppendingString:@".metal"]] encoding:NSUTF8StringEncoding error:&err];
            if(!src) return fail(err.description);
            [source appendString:src];
        }
        MTLCompileOptions *opts=[MTLCompileOptions new];
        // Exact IEEE float semantics: the shaders reproduce the reference's float32 rounding.
        if(@available(macOS 15.0,*)){opts.mathMode=MTLMathModeSafe;opts.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;}
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        library=[d newLibraryWithSource:source options:opts error:&err]; if(!library) return fail(err.description);
    }
    NSArray *names=@[@"scale_raw",@"differences",@"candidates",@"weights",@"green_inputs",@"green_five",@"average_pairs",@"direction_errors",@"direction",@"green_finish",@"chroma_inputs",@"chroma_finish",@"load_tile",@"store_tile",@"dro_subcells",@"dro_histogram",@"dro_smooth",@"dro_weights",@"dro_apply",@"geo_apply",@"sscs_apply",@"look_lm_gamma",@"look_front",@"encoded_front",@"sharp_hblur",@"sharp_apply",@"spica_ycc",@"marble_in",@"marble_blocks",@"marble_sigma",@"marble_binom",@"marble_out",@"post_apply",@"nr_split",@"nr_filter",@"marble_halfc",@"marble_cblocks",@"marble_csigma",@"marble_cup",@"dro_math_test",@"interleave3",@"geo_sscs_apply",@"candidates_both",@"green_inputs_five",@"green_finish_fused",@"dro_smooth_weights",@"marble_out_post",@"load_scale_tile",@"weights_candidates",@"marble_out_packed",@"dro_fix_args",@"dro_apply_fix",@"dro_apply_exact",@"itp_edges",@"itp_final",@"marble_in_half",@"spatial_fused",@"lut_u16"];
    if(names.count!=NPIPE) return fail(@"Metal kernel table size mismatch");
    pipeline_names=names;
    id<MTLComputePipelineState> ps[NPIPE];
    for(int i=0;i<NPIPE;i++) { id<MTLFunction> f=[library newFunctionWithName:names[i]]; if(!f) return fail(@"Missing Metal function"); ps[i]=[d newComputePipelineStateWithFunction:f error:&err]; if(!ps[i]) return fail(err.description); }
    queue=[d newCommandQueue]; if(!queue) return fail(@"Cannot create Metal command queue");
    aux_queue=[d newCommandQueue];if(!aux_queue)return fail(@"Cannot create Metal auxiliary queue");
    grid_event=[d newSharedEvent];if(!grid_event)return fail(@"Cannot create Metal shared event");
    raw_event=[d newSharedEvent];
    if(!raw_event)return fail(@"Cannot create Metal shared event");
    for(int i=0;i<NPIPE;i++) pipelines[i]=ps[i]; device=d; return 0;
}}
static void dispatch(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,Args *args) {
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,kernel); [e setComputePipelineState:pipelines[kernel]];
    for(NSUInteger i=0;i<buffers.count;i++) bind(e,buffers[i],i);
    [e setBytes:args length:sizeof(*args) atIndex:buffers.count];
    [e dispatchThreads:MTLSizeMake(args->width,args->height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; [e endEncoding];
}
int sonylook_metal_itp_debug(const uint16_t *raw, uint32_t w, uint32_t h, const int32_t *black, const float *gains, int32_t scale, const float *blend, int stage, float *out) { @autoreleasepool {
    if(!device || !raw || !out || w<12 || h<12 || (stage<0 || (stage>6 && stage!=18 && stage!=19 && (stage<22 || stage>27)))) return fail(@"Invalid Metal ITP arguments");
    size_t n=(size_t)w*h; Args a={.width=w,.height=h,.scale=scale}; memcpy(a.black,black,sizeof(a.black)); memcpy(a.gains,gains,sizeof(a.gains)); a.blend_k=blend[0]; a.blend_lo=blend[1]; a.blend_hi=blend[2]; a.blend_half=blend[3];
    id<MTLBuffer> input=in_buf(raw,n*2);
    id<MTLBuffer> src=acquire(n*4,MTLResourceStorageModeShared);
    id<MTLBuffer> ah=acquire(n*4,MTLResourceStorageModeShared); id<MTLBuffer> av=acquire(n*4,MTLResourceStorageModeShared);
    id<MTLBuffer> ch=acquire(n*4,MTLResourceStorageModeShared); id<MTLBuffer> cv=acquire(n*4,MTLResourceStorageModeShared);
    id<MTLBuffer> dir=acquire(n*4,MTLResourceStorageModeShared); id<MTLBuffer> bw=acquire(n*4,MTLResourceStorageModeShared);
    if (!input || !src || !ah || !av || !ch || !cv || !dir || !bw) return fail(@"Metal buffer allocation failed");
    id<MTLBuffer> mosaic=nil, ma=nil, mb=nil, ta=nil, tb=nil, five=nil, aa=nil, ab=nil;
    if(stage>=18) {
        mosaic=acquire(n*4,MTLResourceStorageModeShared); ma=acquire(n*4,MTLResourceStorageModeShared); mb=acquire(n*4,MTLResourceStorageModeShared); ta=acquire(n*4,MTLResourceStorageModeShared); tb=acquire(n*4,MTLResourceStorageModeShared); five=acquire(n*4,MTLResourceStorageModeShared); aa=acquire(n*4,MTLResourceStorageModeShared); ab=acquire(n*4,MTLResourceStorageModeShared);
        if(!mosaic || !ma || !mb || !ta || !tb || !five || !aa || !ab) return fail(@"Metal buffer allocation failed");
    }
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1; dispatch(cmd,0,@[input,src],&a);
    if(stage>=1 && stage<=6) dispatch(cmd,1,@[src,ah,av],&a);
    if(stage>=3 && stage<=6) { dispatch(cmd,2,@[ah,ch],&a); dispatch(cmd,2,@[av,cv],&a); }
    if(stage>=4 && stage<=6) dispatch(cmd,3,@[ch,cv,dir,bw],&a);
    if(stage>=18) dispatch(cmd,4,@[src,mosaic,ma,mb,ta,tb],&a);
    if(stage==19) dispatch(cmd,5,@[mosaic,five],&a);
    if(stage==22 || stage==23) dispatch(cmd,6,@[ma,mb,ta,tb,aa,ab],&a);
    [cmd commit]; [cmd waitUntilCompleted]; if(cmd.error) return fail(cmd.error.description);
    id<MTLBuffer> selected = stage==0 ? src : stage==1 ? ah : stage==2 ? av : stage==3 ? ch : stage==6 ? bw : dir;
    if(stage==18) selected=mosaic; else if(stage==19) selected=five; else if(stage==22) selected=aa; else if(stage==23) selected=ab; else if(stage==24) selected=ta; else if(stage==25) selected=tb; else if(stage==26) selected=ma; else if(stage==27) selected=mb;
    memcpy(out,selected.contents,n*4); return done(0);
}}
static void encode_tile(id<MTLCommandBuffer> cmd, id<MTLBuffer> input, NSArray *v, Args *a) {
    dispatch(cmd,0,@[input,v[0]],a);dispatch(cmd,1,@[v[0],v[1],v[2]],a);
    dispatch(cmd,42,@[v[1],v[2],v[3],v[4]],a);dispatch(cmd,3,@[v[3],v[4],v[5],v[6]],a);
    dispatch(cmd,43,@[v[0],v[7],v[8],v[9],v[10],v[11],v[12]],a);
    dispatch(cmd,7,@[v[7],v[1],v[2],v[3],v[4]],a);dispatch(cmd,8,@[v[1],v[2],v[3],v[4],v[15]],a);
    dispatch(cmd,44,@[v[8],v[9],v[10],v[11],v[5],v[6],v[12],v[15],v[16]],a);
    dispatch(cmd,10,@[v[0],v[17],v[18],v[19],v[20]],a);dispatch(cmd,11,@[v[17],v[18],v[19],v[20],v[5],v[16],v[22],v[23],v[24]],a);
}
static void itp_dispatch_encoder(id<MTLComputeCommandEncoder> e,int kernel,NSArray *buffers,Args *a) {
    [e setComputePipelineState:pipelines[kernel]];NSArray *__b=buffers;
    for(NSUInteger __i=0;__i<__b.count;__i++)bind(e,__b[__i],__i);
    [e setBytes:a length:sizeof(*a) atIndex:__b.count];
    [e dispatchThreads:MTLSizeMake(a->width,a->height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
}
static void encode_tile_scaled(id<MTLCommandBuffer> cmd, id source, NSArray *v, Args *a) {
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,100);
    if(source){itp_dispatch_encoder(e,47,@[source,v[0]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
    itp_dispatch_encoder(e,1,@[v[0],v[1],v[2]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,48,@[v[1],v[2],v[5],v[6]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,43,@[v[0],v[7],v[8],v[9],v[10],v[11],v[12]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,7,@[v[7],v[1],v[2],v[3],v[4]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,8,@[v[1],v[2],v[3],v[4],v[15]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,44,@[v[8],v[9],v[10],v[11],v[5],v[6],v[12],v[15],v[16]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,10,@[v[0],v[17],v[18],v[19],v[20]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,11,@[v[17],v[18],v[19],v[20],v[5],v[16],v[22],v[23],v[24]],a);
    [e endEncoding];
}
// Default ITP: three passes per tile (see itp_final in itp.metal); the last
// writes the tile's RGB straight into the image planes.
static void encode_tile_three(id<MTLCommandBuffer> cmd,id source,NSArray *v,Args *a,id R,id G,id B) {
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,101);
    itp_dispatch_encoder(e,47,@[source,v[0]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    itp_dispatch_encoder(e,53,@[v[0],v[1],v[2],v[13],v[14],v[3],v[4]],a);[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    NSArray *fin=@[v[0],v[1],v[2],v[13],v[14],v[3],v[4],R,G,B];
    [e setComputePipelineState:pipelines[54]];
    for(NSUInteger i=0;i<fin.count;i++)bind(e,fin[i],i);
    [e setBytes:a length:sizeof(*a) atIndex:fin.count];
    [e dispatchThreads:MTLSizeMake(a->tile_width,a->tile_height,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
    [e endEncoding];
}
int sonylook_metal_itp_tile(const uint16_t *raw, uint32_t w, uint32_t h, const int32_t *black, const float *gains, int32_t scale, const float *params, uint16_t *r, uint16_t *g, uint16_t *b) { @autoreleasepool {
    if(!device || !raw || !r || !g || !b || w<12 || h<12) return fail(@"Invalid Metal ITP arguments");
    size_t n=(size_t)w*h; Args a={.width=w,.height=h,.scale=scale}; memcpy(a.black,black,sizeof(a.black)); memcpy(a.gains,gains,sizeof(a.gains));
    a.blend_k=params[0];a.blend_lo=params[1];a.blend_hi=params[2];a.blend_half=params[3];a.k1=params[4];a.k2=params[5];a.k3=params[6];a.k4=params[7];a.k5=params[8];a.k6=params[9];
    id<MTLBuffer> input=in_buf(raw,n*2);
    NSMutableArray *v=[NSMutableArray arrayWithCapacity:25];
    for(int i=0;i<25;i++){id<MTLBuffer> buf=acquire(n*(i>=22?2:4),MTLResourceStorageModeShared);if(!buf)return fail(@"Metal buffer allocation failed");[v addObject:buf];}
    if(!input)return fail(@"Metal input allocation failed");
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1;
    encode_tile(cmd,input,v,&a);
    [cmd commit];[cmd waitUntilCompleted];if(cmd.error)return fail(cmd.error.description);
    memcpy(r,((id<MTLBuffer>)v[22]).contents,n*2);memcpy(g,((id<MTLBuffer>)v[23]).contents,n*2);memcpy(b,((id<MTLBuffer>)v[24]).contents,n*2);return done(0);
}}

// Whole image: scratch planes are allocated once and reused in GPU command order.
int sonylook_metal_itp_image(const uint16_t *raw, uint32_t w, uint32_t h, const int32_t *black, const float *gains, int32_t scale, const float *params, uint32_t tile, uint32_t halo, const NrArgs *nr_args, const int32_t *nr_thr, const RawStream *stream, uint16_t *r, uint16_t *g, uint16_t *b) { @autoreleasepool {
    if(!device || !raw || !r || !g || !b || w<12 || h<12 || tile==0 || halo<10) return fail(@"Invalid Metal ITP image arguments");
    size_t count=(size_t)w*h;
    uint32_t maxw=MIN(w,tile)+2*halo, maxh=MIN(h,tile)+2*halo;
    size_t tilecount=(size_t)maxw*maxh;
    // Streaming: RawNR and ITP read the w x h crop at (src_x, src_y) of the
    // GPU-resident sensor raw in place, as its rows are decoded; `raw` is unused
    // and nothing is read on the host.
    id image=nil;
    uint32_t src_stride=w,src_origin=0;
    if(stream){
        if(!stream->src||stream->src_x+w>stream->src_stride||stream->src_y+h>stream->src_rows)
            return fail(@"Streamed ITP needs an in-bounds crop of the sensor raw");
        image=find_resident(stream->src,(size_t)stream->src_stride*stream->src_rows*2);
        if(!image)return fail(@"Streamed ITP needs a GPU-resident sensor raw");
        src_stride=stream->src_stride;src_origin=stream->src_y*stream->src_stride+stream->src_x;
    } else image=in_buf(raw,count*2);
    id<MTLBuffer> input=acquire(tilecount*2,MTLResourceStorageModePrivate);
    id<MTLBuffer> nr_image=nil,nr_hp=nil,nr_base=nil,nr_thr_buf=nil;
    NrArgs nr_local={0};
    id red=out_buf(r,count*2),green=out_buf(g,count*2),blue=out_buf(b,count*2);
    if(!image || !input || !red || !green || !blue) return fail(@"Metal image buffer allocation failed");
    NSMutableArray *v=[NSMutableArray arrayWithCapacity:25];
    for(int i=0;i<25;i++){id<MTLBuffer> buf=acquire(tilecount*(i>=22?2:4),MTLResourceStorageModePrivate);if(!buf)return fail(@"Metal scratch allocation failed");[v addObject:buf];}
    if(nr_args) {
        nr_image=acquire(count*2,MTLResourceStorageModePrivate);
        nr_hp=acquire(count*2,MTLResourceStorageModePrivate);
        nr_base=acquire(count*2,MTLResourceStorageModePrivate);
        nr_thr_buf=in_buf(nr_thr,3*0x8000*4);
        nr_local=*nr_args;nr_local.width=w;nr_local.height=h;
        nr_local.raw_stride=src_stride;nr_local.raw_origin=src_origin;
        if(!nr_image||!nr_hp||!nr_base||!nr_thr_buf)return fail(@"Metal fused RawNR buffers failed");
    }
    Args a={.image_width=w,.image_height=h,.halo=halo,.scale=scale};
    // ITP reads the raw view directly, or RawNR's dense output.
    if(!nr_args){a.src_stride=src_stride;a.src_origin=src_origin;}memcpy(a.black,black,sizeof(a.black));memcpy(a.gains,gains,sizeof(a.gains));
    a.blend_k=params[0];a.blend_lo=params[1];a.blend_hi=params[2];a.blend_half=params[3];a.k1=params[4];a.k2=params[5];a.k3=params[6];a.k4=params[7];a.k5=params[8];a.k6=params[9];
    id<MTLCommandBuffer> cmd=command_buffer();
    // SONYLOOK_ITP_FUSED: unset/2 = three passes, 1 = the nine-kernel encoder, 0 = separate kernels.
    const char *fused_env=getenv("SONYLOOK_ITP_FUSED");
    int itp_mode=fused_env?atoi(fused_env):2;
    // Row frontiers: before each row of tiles, make sure its input rows exist.
    // ITP reads rows [y - halo, y + tile + halo) (mirrored inside the image);
    // nr_filter row r reads base/hp rows r +- 8, nr_split row r raw rows r +- 2.
    uint32_t raw_ready=stream?0:h,split_done=0,filter_done=0;
    for(uint32_t y=0;y<h;y+=tile) {
        uint32_t need=MIN(h,y+MIN(tile,h-y)+halo);
        uint32_t need_split=nr_args?MIN(h,need+8):0,need_raw=nr_args?MIN(h,need_split+2):need;
        if(raw_ready<need_raw){
            [cmd encodeWaitForEvent:raw_event value:stream->base+stream->src_y+need_raw];
            raw_ready=need_raw;
        }
        if(nr_args){
            encode_band(cmd,33,@[image,nr_hp,nr_base],&nr_local,sizeof(nr_local),w,split_done,need_split-split_done);
            split_done=need_split;
            encode_band(cmd,34,@[image,nr_hp,nr_base,nr_thr_buf,nr_image],&nr_local,sizeof(nr_local),w,filter_done,need-filter_done);
            filter_done=need;
        }
    for(uint32_t x=0;x<w;x+=tile) {
        a.tile_x=x;a.tile_y=y;a.tile_width=MIN(tile,w-x);a.tile_height=MIN(tile,h-y);
        a.width=a.tile_width+2*halo;a.height=a.tile_height+2*halo;
        id<MTLCommandBuffer> tile_cmd=cmd;
        if(itp_mode==2) {
            encode_tile_three(tile_cmd,nr_args?nr_image:image,v,&a,red,green,blue);
            continue;
        }
        if(nr_args) {
            encode_tile_scaled(tile_cmd,nr_image,v,&a);
        } else if(itp_mode) {
            encode_tile_scaled(tile_cmd,image,v,&a);
        } else {
            dispatch(tile_cmd,12,@[image,input],&a);
            dispatch(tile_cmd,0,@[input,v[0]],&a);
            encode_tile(tile_cmd,input,v,&a);
        }
        dispatch(tile_cmd,13,@[v[22],v[23],v[24],red,green,blue],&a);
    }}
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(red,r,count*2);copy_back(green,g,count*2);copy_back(blue,b,count*2);return done(0);
}}

typedef struct {
    uint32_t width,height,raw_width,raw_height;
    int32_t origin_x,origin_y;
    uint32_t subpx_x,subpx_y,org_x,org_y,per_x,per_y;
    int32_t black_r,black_g,black_b;
    float log2_max,strength,k_up,k_down,scale;
    uint32_t fix_capacity;
    uint32_t src_stride,src_x,src_y;
} DroArgs;
static void dispatch_dro(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,DroArgs *args, uint32_t w,uint32_t h) {
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,kernel);[e setComputePipelineState:pipelines[kernel]];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:args length:sizeof(*args) atIndex:buffers.count];
    [e dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
}
static void dispatch_dro(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,DroArgs *args, uint32_t w,uint32_t h);
// DRO apply with the fast exp2 (see dro.metal): main pass, indirect sizes,
// exact recompute of the listed pixels, and an exact full pass that only gets
// threadgroups when the list overflowed. No CPU round trip, so it batches.
static int encode_dro_apply(id<MTLCommandBuffer> cmd,NSArray *io,DroArgs *a,uint32_t w,uint32_t h) {
    size_t n=(size_t)w*h;uint32_t cap=(uint32_t)MAX((size_t)4096,n/64);a->fix_capacity=cap;
    id<MTLBuffer> list=acquire((size_t)cap*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> count=acquire(4,MTLResourceStorageModeShared);
    id<MTLBuffer> ind=acquire(6*4,MTLResourceStorageModePrivate);
    if(!list||!count||!ind)return fail(@"Metal DRO list allocation failed");
    memset(count.contents,0,4);
    NSArray *all=[io arrayByAddingObjectsFromArray:@[list,count]];
    dispatch_dro(cmd,18,all,a,w,h);
    dispatch_dro(cmd,50,@[count,ind],a,1,1);
    for(int pass=0;pass<2;pass++){
        id<MTLComputeCommandEncoder> e=new_encoder(cmd,(pass?52:51));[e setComputePipelineState:pipelines[pass?52:51]];
        for(NSUInteger i=0;i<all.count;i++)bind(e,all[i],i);
        [e setBytes:a length:sizeof(*a) atIndex:all.count];
        [e dispatchThreadgroupsWithIndirectBuffer:ind indirectBufferOffset:pass?12:0
                           threadsPerThreadgroup:pass?MTLSizeMake(16,16,1):MTLSizeMake(64,1,1)];
        [e endEncoding];
    }
    return 0;
}
static DroArgs make_dro_args(uint32_t raww,uint32_t rawh,uint32_t w,uint32_t h,const int32_t *black,
                             int32_t ox,int32_t oy,float log2_max,float strength,
                             float k_up,float k_down,float scale) {
    DroArgs a={.width=w,.height=h,.raw_width=raww,.raw_height=rawh,.origin_x=ox,.origin_y=oy,
               .black_r=black[0],.black_g=black[1],.black_b=black[2],.log2_max=log2_max,
               .strength=strength,.k_up=k_up,.k_down=k_down,.scale=scale};
    a.subpx_x=((raww>>2)/112)*4;a.subpx_y=((rawh>>2)/84)*4;a.per_x=a.subpx_x/4;a.per_y=a.subpx_y/4;
    a.org_x=((raww-a.subpx_x*112)>>1)&~1u;a.org_y=((rawh-a.subpx_y*84)>>1)&~1u;
    a.src_stride=raww;a.src_x=0;a.src_y=0;
    return a;
}
// The grid from a raww x rawh crop at (src_x, src_y) of a GPU-resident
// src_stride-wide array, waiting on the aux queue for raw_event >= wait (0: none).
int sonylook_metal_dro_grid_view(const uint16_t *src,uint32_t src_stride,uint32_t src_rows,uint32_t src_x,uint32_t src_y,uint64_t wait,
                                 uint32_t raww,uint32_t rawh,const int32_t *black,
                                 int32_t ox,int32_t oy,const float *logtab,float log2_max,float strength,
                                 float k_up,float k_down,float scale,float *weights,float *sums);
int sonylook_metal_dro_grid(const uint16_t *raw,uint32_t raww,uint32_t rawh,const int32_t *black,
                            int32_t ox,int32_t oy,const float *logtab,const float *curve,
                            float log2_max,float strength,float k_up,float k_down,float scale,
                            float *weights,float *sums) {
    (void)curve;
    return sonylook_metal_dro_grid_view(raw,raww,rawh,0,0,0,raww,rawh,black,ox,oy,logtab,log2_max,strength,k_up,k_down,scale,weights,sums);
}
int sonylook_metal_dro_grid_view(const uint16_t *src,uint32_t src_stride,uint32_t src_rows,uint32_t src_x,uint32_t src_y,uint64_t wait,
                                 uint32_t raww,uint32_t rawh,const int32_t *black,
                                 int32_t ox,int32_t oy,const float *logtab,float log2_max,float strength,
                                 float k_up,float k_down,float scale,float *weights,float *sums) { @autoreleasepool {
    if(!device||!aux_queue||!grid_event||!src||!weights||!sums||raww<448||rawh<336||
       src_x+raww>src_stride||src_y+rawh>src_rows)return fail(@"Invalid Metal DRO grid arguments");
    DroArgs a=make_dro_args(raww,rawh,1,1,black,ox,oy,log2_max,strength,k_up,k_down,scale);
    a.src_stride=src_stride;a.src_x=src_x;a.src_y=src_y;
    // The grid finishes asynchronously on aux_queue. W/S are read by a later
    // call, so they must be resident arrays (metal_itp.empty), never pooled
    // copies that would need a copy-back.
    id Wb=find_resident(weights,8*6*14*4),Sb=find_resident(sums,8*6*14*4);
    if(!Wb||!Sb)return fail(@"Metal DRO grid outputs must be GPU-resident arrays");
    NSUInteger mark;@synchronized(buf_lock){mark=call_bufs.count;}
    size_t srclen=(size_t)src_stride*src_rows*2;
    // A wait means the source is still being produced on the GPU: it must be
    // read in place, never copied now.
    id rawbuf=wait?find_resident(src,srclen):in_buf(src,srclen),tab=in_buf(logtab,2049*4);
    if(wait&&!rawbuf)return fail(@"Streamed DRO grid needs a GPU-resident source");
    id sub=acquire(112*84*4,MTLResourceStorageModeShared),hist=acquire(8*6*14*4,MTLResourceStorageModePrivate);
    if(!rawbuf||!tab||!sub||!hist)return fail(@"Metal DRO grid allocation failed");
    // Temporaries live until the aux command buffer completes, not until the
    // next main-queue call's done(): take them out of call_bufs.
    NSArray *grid_bufs;
    @synchronized(buf_lock) {
        NSRange r=NSMakeRange(mark,call_bufs.count-mark);
        grid_bufs=[call_bufs subarrayWithRange:r];[call_bufs removeObjectsInRange:r];
    }
    id<MTLCommandBuffer> cmd=[aux_queue commandBuffer];
    if(wait)[cmd encodeWaitForEvent:raw_event value:wait];
    dispatch_dro(cmd,14,@[rawbuf,tab,sub],&a,112,84);
    dispatch_dro(cmd,15,@[sub,hist],&a,8,6);
    dispatch_dro(cmd,45,@[hist,Wb,Sb],&a,8,6);
    pending_grid_value=++grid_event_value;
    [cmd encodeSignalEvent:grid_event value:pending_grid_value];
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> c){(void)c;for(id<MTLBuffer> b in grid_bufs)pool_put(b);}];
    [cmd commit];
    return 0;
}}
int sonylook_metal_dro_grid_apply(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,
                                  uint32_t raww,uint32_t rawh,const int32_t *black,int32_t ox,int32_t oy,
                                  const float *logtab,const float *curve,float log2_max,float strength,
                                  float k_up,float k_down,float scale,const float *weights,const float *sums,
                                  uint16_t *ro,uint16_t *go,uint16_t *bo) { @autoreleasepool {
    if(!device||!r||!g||!b||!ro||!go||!bo||!weights||!sums||!pending_grid_value)return fail(@"Invalid prepared Metal DRO arguments");
    DroArgs a=make_dro_args(raww,rawh,w,h,black,ox,oy,log2_max,strength,k_up,k_down,scale);
    size_t n=(size_t)w*h;
    id rb=in_buf(r,n*2),gb=in_buf(g,n*2),bb=in_buf(b,n*2),tab=in_buf(logtab,2049*4),cv=in_buf(curve,104*4);
    id Wb=in_buf(weights,8*6*14*4),Sb=in_buf(sums,8*6*14*4);
    id R=out_buf(ro,n*2),G=out_buf(go,n*2),B=out_buf(bo,n*2);
    if(!rb||!gb||!bb||!tab||!cv||!Wb||!Sb||!R||!G||!B)return fail(@"Prepared Metal DRO allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();
    [cmd encodeWaitForEvent:grid_event value:pending_grid_value];
    if(encode_dro_apply(cmd,@[rb,gb,bb,Wb,Sb,cv,tab,R,G,B],&a,w,h))return -1;
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(R,ro,n*2);copy_back(G,go,n*2);copy_back(B,bo,n*2);
    return done(0);
}}
int sonylook_metal_dro(const uint16_t *raw,uint32_t raww,uint32_t rawh,const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,const int32_t *black,int32_t ox,int32_t oy,const float *logtab,const float *curve,float log2_max,float strength,float k_up,float k_down,float scale,uint16_t *ro,uint16_t *go,uint16_t *bo,float *debug_sub,float *debug_w,float *debug_s) { @autoreleasepool {
    if(!device||!raw||!r||!g||!b||!ro||!go||!bo||raww<448||rawh<336||!w||!h)return fail(@"Invalid Metal DRO arguments");
    DroArgs a=make_dro_args(raww,rawh,w,h,black,ox,oy,log2_max,strength,k_up,k_down,scale);
    size_t n=(size_t)w*h;
    id<MTLBuffer> rawbuf=in_buf(raw,(size_t)raww*rawh*2);
    id<MTLBuffer> rb=in_buf(r,n*2);id<MTLBuffer> gb=in_buf(g,n*2);id<MTLBuffer> bb=in_buf(b,n*2);
    id<MTLBuffer> tab=in_buf(logtab,2049*4);id<MTLBuffer> cv=in_buf(curve,104*4);
    id<MTLBuffer> sub=acquire(112*84*4,MTLResourceStorageModeShared);id<MTLBuffer> hist=acquire(8*6*14*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> W=acquire(8*6*14*4,MTLResourceStorageModeShared);id<MTLBuffer> S=acquire(8*6*14*4,MTLResourceStorageModeShared);
    id R=out_buf(ro,n*2),G=out_buf(go,n*2),B=out_buf(bo,n*2);
    if(!rawbuf||!rb||!gb||!bb||!tab||!cv||!sub||!hist||!W||!S||!R||!G||!B)return fail(@"Metal DRO buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();
    dispatch_dro(cmd,14,@[rawbuf,tab,sub],&a,112,84);dispatch_dro(cmd,15,@[sub,hist],&a,8,6);dispatch_dro(cmd,45,@[hist,W,S],&a,8,6);
    if(encode_dro_apply(cmd,@[rb,gb,bb,W,S,cv,tab,R,G,B],&a,w,h))return -1;
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(R,ro,n*2);copy_back(G,go,n*2);copy_back(B,bo,n*2);
    if(!batch_cmd) {
        if(debug_sub)memcpy(debug_sub,sub.contents,112*84*4);if(debug_w)memcpy(debug_w,W.contents,8*6*14*4);if(debug_s)memcpy(debug_s,S.contents,8*6*14*4);
    }
    return done(0);
}}

typedef struct {
    uint32_t width,height,output_width,output_height;
    int32_t origin_x,origin_y,region_x,region_y;
    float cx,cy,s,s2,step;
    uint32_t n;
    float off[6];
} GeoArgs;
int sonylook_metal_geo(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,
                       int32_t ox,int32_t oy,int32_t rx,int32_t ry,uint32_t ow,uint32_t oh,
                       const float *lens,const float *tr,const float *tg,const float *tb,uint32_t n,
                       uint16_t *ro,uint16_t *go,uint16_t *bo) { @autoreleasepool {
    if(!device||!r||!g||!b||!ro||!go||!bo||!w||!h||!ow||!oh||n<2)return fail(@"Invalid Metal GeoCorr arguments");
    GeoArgs a={.width=w,.height=h,.output_width=ow,.output_height=oh,.origin_x=ox,.origin_y=oy,.region_x=rx,.region_y=ry,
               .cx=lens[0],.cy=lens[1],.s=lens[2],.s2=lens[3],.step=lens[4],.n=n};
    memcpy(a.off,lens+5,sizeof(a.off));
    size_t count=(size_t)w*h,outcount=(size_t)ow*oh;
    id<MTLBuffer> rb=in_buf(r,count*2);
    id<MTLBuffer> gb=in_buf(g,count*2);
    id<MTLBuffer> bb=in_buf(b,count*2);
    id<MTLBuffer> rt=in_buf(tr,n*4);
    id<MTLBuffer> gt=in_buf(tg,n*4);
    id<MTLBuffer> bt=in_buf(tb,n*4);
    id R=out_buf(ro,outcount*2),G=out_buf(go,outcount*2),B=out_buf(bo,outcount*2);
    if(!rb||!gb||!bb||!rt||!gt||!bt||!R||!G||!B)return fail(@"Metal GeoCorr buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();id<MTLComputeCommandEncoder> e=new_encoder(cmd,19);
    [e setComputePipelineState:pipelines[19]];NSArray *buffers=@[rb,gb,bb,rt,gt,bt,R,G,B];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:&a length:sizeof(a) atIndex:buffers.count];
    [e dispatchThreads:MTLSizeMake(ow,oh,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(R,ro,outcount*2);copy_back(G,go,outcount*2);copy_back(B,bo,outcount*2);return done(0);
}}

typedef struct { uint32_t width,height; int32_t t,tr,tg,tb,wr,wg,wb; } SscsArgs;
int sonylook_metal_sscs(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,const int32_t *params,uint16_t *out) { @autoreleasepool {
    if(!device||!r||!g||!b||!w||!h||!params||!out)return fail(@"Invalid Metal SSCS arguments");
    SscsArgs a={.width=w,.height=h,.t=params[0],.tr=params[1],.tg=params[2],.tb=params[3],.wr=params[4],.wg=params[5],.wb=params[6]};
    size_t n=(size_t)w*h;
    id<MTLBuffer> rb=in_buf(r,n*2);
    id<MTLBuffer> gb=in_buf(g,n*2);
    id<MTLBuffer> bb=in_buf(b,n*2);
    id result=out_buf(out,n*6);
    if(!rb||!gb||!bb||!result)return fail(@"Metal SSCS buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();id<MTLComputeCommandEncoder> e=new_encoder(cmd,20);
    [e setComputePipelineState:pipelines[20]];NSArray *buffers=@[rb,gb,bb,result];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:&a length:sizeof(a) atIndex:buffers.count];
    [e dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(result,out,n*6);return done(0);
}}

typedef struct { GeoArgs geo; SscsArgs sscs; } GeoSscsArgs;
// GeoCorr and SSCS fused: sampled uint16 values feed SSCS in the same kernel,
// and the final result is written directly as interleaved RGB.
int sonylook_metal_geo_sscs(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,
                            int32_t ox,int32_t oy,int32_t rx,int32_t ry,uint32_t ow,uint32_t oh,
                            const float *lens,const float *tr,const float *tg,const float *tb,uint32_t n,
                            const int32_t *params,uint16_t *out) { @autoreleasepool {
    if(!device||!r||!g||!b||!out||!w||!h||!ow||!oh||n<2||!params)return fail(@"Invalid Metal GeoCorr+SSCS arguments");
    GeoSscsArgs a={.geo={.width=w,.height=h,.output_width=ow,.output_height=oh,.origin_x=ox,.origin_y=oy,
                         .region_x=rx,.region_y=ry,.cx=lens[0],.cy=lens[1],.s=lens[2],.s2=lens[3],.step=lens[4],.n=n},
                   .sscs={.width=ow,.height=oh,.t=params[0],.tr=params[1],.tg=params[2],.tb=params[3],
                          .wr=params[4],.wg=params[5],.wb=params[6]}};
    memcpy(a.geo.off,lens+5,sizeof(a.geo.off));
    size_t count=(size_t)w*h,outcount=(size_t)ow*oh;
    id<MTLBuffer> rb=in_buf(r,count*2);
    id<MTLBuffer> gb=in_buf(g,count*2);
    id<MTLBuffer> bb=in_buf(b,count*2);
    id<MTLBuffer> rt=in_buf(tr,n*4);
    id<MTLBuffer> gt=in_buf(tg,n*4);
    id<MTLBuffer> bt=in_buf(tb,n*4);
    id result=out_buf(out,outcount*6);
    if(!rb||!gb||!bb||!rt||!gt||!bt||!result)return fail(@"Metal GeoCorr+SSCS buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();id<MTLComputeCommandEncoder> e=new_encoder(cmd,41);
    [e setComputePipelineState:pipelines[41]];NSArray *buffers=@[rb,gb,bb,rt,gt,bt,result];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:&a length:sizeof(a) atIndex:buffers.count];
    [e dispatchThreads:MTLSizeMake(ow,oh,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(result,out,outcount*6);return done(0);
}}

typedef struct { uint32_t width,height; float thresholds[3]; } LookArgs;
int sonylook_metal_look(const uint16_t *rgb,uint32_t w,uint32_t h,const float *table,const int16_t *gamma,const float *atan,const float *thresholds,int16_t *out) { @autoreleasepool {
    if(!device||!rgb||w<5||!h||!table||!gamma||!atan||!thresholds||!out)return fail(@"Invalid Metal Look arguments");
    size_t n=(size_t)w*h;LookArgs a={.width=w,.height=h};memcpy(a.thresholds,thresholds,sizeof(a.thresholds));
    id<MTLBuffer> rgbb=in_buf(rgb,n*6);
    id<MTLBuffer> tab=in_buf(table,1024*18*4);
    id<MTLBuffer> gb=in_buf(gamma,65536*2);
    id<MTLBuffer> ab=in_buf(atan,512*4);
    id result=out_buf(out,n*6);
    if(!rgbb||!tab||!gb||!ab||!result)return fail(@"Metal Look buffer allocation failed");
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1;id<MTLComputeCommandEncoder> e=new_encoder(cmd,21);
    [e setComputePipelineState:pipelines[21]];NSArray *buffers=@[rgbb,tab,gb,ab,result];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:&a length:sizeof(a) atIndex:buffers.count];[e dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
    [cmd commit];[cmd waitUntilCompleted];if(cmd.error)return fail(cmd.error.description);copy_back(result,out,n*6);return done(0);
}}

typedef struct { uint32_t width,height; float thresholds[3]; int32_t c0,c1,c2; float k8,k9,k10,k11,gcr_pos,gcr_neg,gcb_pos,gcb_neg; float area[16]; int32_t area_flag,w9,w12,w8,w11; } FrontArgs;
static int run_front(int kernel,NSArray *inputs,uint32_t w,uint32_t h,const FrontArgs *args,uint16_t *y,uint16_t *cr,uint16_t *cb) {
    size_t n=(size_t)w*h;FrontArgs a=*args;a.width=w;a.height=h;
    id yb=out_buf(y,n*2),crb=out_buf(cr,n*2),cbb=out_buf(cb,n*2);
    if(!yb||!crb||!cbb)return fail(@"Metal front buffer allocation failed");
    NSArray *buffers=[inputs arrayByAddingObjectsFromArray:@[yb,crb,cbb]];
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1;id<MTLComputeCommandEncoder> e=new_encoder(cmd,22);
    [e setComputePipelineState:pipelines[kernel]];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:&a length:sizeof(a) atIndex:buffers.count];[e dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
    [cmd commit];[cmd waitUntilCompleted];if(cmd.error)return fail(cmd.error.description);
    copy_back(yb,y,n*2);copy_back(crb,cr,n*2);copy_back(cbb,cb,n*2);return done(0);
}
// LinearMatrix16, MainGamma, RGB2YCC, AreaComp and ChromaSuppress in one pass.
int sonylook_metal_front(const uint16_t *rgb,uint32_t w,uint32_t h,const float *table,const int16_t *gamma,const float *atan,const FrontArgs *args,uint16_t *y,uint16_t *cr,uint16_t *cb) { @autoreleasepool {
    if(!device||!rgb||w<5||!h||!table||!gamma||!atan||!args||!y||!cr||!cb)return fail(@"Invalid Metal front arguments");
    size_t n=(size_t)w*h;
    id<MTLBuffer> rgbb=in_buf(rgb,n*6);
    id<MTLBuffer> tab=in_buf(table,1024*18*4);
    id<MTLBuffer> gb=in_buf(gamma,65536*2);
    id<MTLBuffer> ab=in_buf(atan,512*4);
    if(!rgbb||!tab||!gb||!ab)return fail(@"Metal front buffer allocation failed");
    return run_front(22,@[rgbb,tab,gb,ab],w,h,args,y,cr,cb);
}}
// RGB2YCC, AreaComp and ChromaSuppress on interleaved int16 MainGamma output.
int sonylook_metal_encoded_front(const int16_t *enc,uint32_t w,uint32_t h,const FrontArgs *args,uint16_t *y,uint16_t *cr,uint16_t *cb) { @autoreleasepool {
    if(!device||!enc||!w||!h||!args||!y||!cr||!cb)return fail(@"Invalid Metal YCC arguments");
    id<MTLBuffer> eb=in_buf(enc,(size_t)w*h*6);
    if(!eb)return fail(@"Metal YCC buffer allocation failed");
    return run_front(23,@[eb],w,h,args,y,cr,cb);
}}

typedef struct { uint32_t width,height; float sharp_thr,sharp_s0,sharp_s1,sharp_s9,sharp_k; float yg_s0,yg_s1,yg_s2,yg_s3; uint32_t spica_thr; int32_t spica_full,spica_gain_on; float spica_s,spica_k; float s[25]; } SpatialArgs;
static void encode_grid(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,const void *args,size_t len,uint32_t w,uint32_t h) {
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,kernel);[e setComputePipelineState:pipelines[kernel]];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:args length:len atIndex:buffers.count];[e dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
}
// encode_grid over rows [row0, row0 + rows) of a w-wide image; the kernel
// takes the band as the constant after `args`.
static void encode_band(id<MTLCommandBuffer> cmd,int kernel,NSArray *buffers,const void *args,size_t len,uint32_t w,uint32_t row0,uint32_t rows) {
    if(!rows)return;
    uint32_t band[2]={row0,rows};
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,kernel);[e setComputePipelineState:pipelines[kernel]];
    for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
    [e setBytes:args length:len atIndex:buffers.count];[e setBytes:band length:sizeof(band) atIndex:buffers.count+1];
    [e dispatchThreads:MTLSizeMake(w,rows,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
}
static NSArray *new_buffers(size_t count,size_t length,MTLResourceOptions options) {
    NSMutableArray *v=[NSMutableArray array];
    for(size_t i=0;i<count;i++){id<MTLBuffer> b=acquire(length,options);if(!b)return nil;[v addObject:b];}
    return v;
}
// Sharpness, YGamma, Spica and YCC2RGB from Y/Cr/Cb buffers into rgb[0..2].
static int encode_spatial(id<MTLCommandBuffer> cmd,id<MTLBuffer> y,id<MTLBuffer> cr,id<MTLBuffer> cb,uint32_t w,uint32_t h,const SpatialArgs *args,const int16_t *ylut,const int32_t *patterns,const float *coefs,NSArray *rgb) {
    SpatialArgs a=*args;a.width=w;a.height=h;size_t n=(size_t)w*h;
    id<MTLBuffer> lut=in_buf(ylut,65536*2);
    id<MTLBuffer> pat=in_buf(patterns,256*3*4);
    id<MTLBuffer> cof=in_buf(coefs,100*26*4);
    if(!lut||!pat||!cof)return fail(@"Metal spatial buffer allocation failed");
    // SONYLOOK_SPATIAL_FUSED=0: the three separate kernels with hb/yg planes.
    const char *fused=getenv("SONYLOOK_SPATIAL_FUSED");
    if(!fused||atoi(fused)) {
        id<MTLComputeCommandEncoder> e=new_encoder(cmd,56);[e setComputePipelineState:pipelines[56]];
        NSArray *bufs=[@[y,cr,cb,lut,pat,cof] arrayByAddingObjectsFromArray:rgb];
        for(NSUInteger i=0;i<bufs.count;i++)bind(e,bufs[i],i);
        [e setBytes:&a length:sizeof(a) atIndex:bufs.count];
        [e dispatchThreadgroups:MTLSizeMake((w+31)/32,(h+31)/32,1) threadsPerThreadgroup:MTLSizeMake(32,8,1)];[e endEncoding];
        return 0;
    }
    id<MTLBuffer> hb=acquire((size_t)(w-4)*(h-4)*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> yg=acquire(n*2,MTLResourceStorageModePrivate);
    if(!hb||!yg)return fail(@"Metal spatial buffer allocation failed");
    encode_grid(cmd,24,@[y,hb],&a,sizeof(a),w-4,h-4);
    encode_grid(cmd,25,@[y,hb,lut,yg],&a,sizeof(a),w,h);
    encode_grid(cmd,26,[@[yg,cr,cb,pat,cof] arrayByAddingObjectsFromArray:rgb],&a,sizeof(a),w,h);
    return 0;
}
static NSArray *out3(uint16_t *r,uint16_t *g,uint16_t *b,size_t len) {
    id x=out_buf(r,len),y=out_buf(g,len),z=out_buf(b,len);return (x&&y&&z)?@[x,y,z]:nil;
}
static int finish_rgb(id<MTLCommandBuffer> cmd,NSArray *rgb,size_t n,uint16_t *r,uint16_t *g,uint16_t *b) {
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(rgb[0],r,n*2);copy_back(rgb[1],g,n*2);copy_back(rgb[2],b,n*2);return done(0);
}
int sonylook_metal_spatial(const uint16_t *y,const uint16_t *cr,const uint16_t *cb,uint32_t w,uint32_t h,const SpatialArgs *args,const int16_t *ylut,const int32_t *patterns,const float *coefs,uint16_t *r,uint16_t *g,uint16_t *b) { @autoreleasepool {
    if(!device||!y||!cr||!cb||w<7||h<7||!args||!ylut||!patterns||!coefs||!r||!g||!b)return fail(@"Invalid Metal spatial arguments");
    size_t n=(size_t)w*h;
    id<MTLBuffer> yb=in_buf(y,n*2);
    id<MTLBuffer> crb=in_buf(cr,n*2);
    id<MTLBuffer> cbb=in_buf(cb,n*2);
    NSArray *rgb=out3(r,g,b,n*2);
    if(!yb||!crb||!cbb||!rgb)return fail(@"Metal spatial buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();
    if(encode_spatial(cmd,yb,crb,cbb,w,h,args,ylut,patterns,coefs,rgb))return -1;
    return finish_rgb(cmd,rgb,n,r,g,b);
}}
// LinearMatrix16 through YCC2RGB in one command buffer; Y/Cr/Cb stay on the GPU.
int sonylook_metal_front_spatial(const uint16_t *rgbin,uint32_t w,uint32_t h,const float *table,const int16_t *gamma,const float *atan,const FrontArgs *front,const SpatialArgs *args,const int16_t *ylut,const int32_t *patterns,const float *coefs,uint16_t *r,uint16_t *g,uint16_t *b) { @autoreleasepool {
    if(!device||!rgbin||w<7||h<7||!table||!gamma||!atan||!front||!args||!ylut||!patterns||!coefs||!r||!g||!b)return fail(@"Invalid Metal front/spatial arguments");
    size_t n=(size_t)w*h;FrontArgs fa=*front;fa.width=w;fa.height=h;
    id<MTLBuffer> rgbb=in_buf(rgbin,n*6);
    id<MTLBuffer> tab=in_buf(table,1024*18*4);
    id<MTLBuffer> gb=in_buf(gamma,65536*2);
    id<MTLBuffer> ab=in_buf(atan,512*4);
    NSArray *ycc=new_buffers(3,n*2,MTLResourceStorageModePrivate),*rgb=out3(r,g,b,n*2);
    if(!rgbb||!tab||!gb||!ab||!ycc||!rgb)return fail(@"Metal front/spatial buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();
    encode_grid(cmd,22,[@[rgbb,tab,gb,ab] arrayByAddingObjectsFromArray:ycc],&fa,sizeof(fa),w,h);
    if(encode_spatial(cmd,ycc[0],ycc[1],ycc[2],w,h,args,ylut,patterns,coefs,rgb))return -1;
    return finish_rgb(cmd,rgb,n,r,g,b);
}}

typedef struct { uint32_t width,height,capacity; int32_t effect,conv,m_fwd[9],m_inv[9]; float t_r,t_b,c_hi[3],c_lo[3]; int32_t tone,shade,cw,ch; float g_hi[11],g_lo[11],d_hi[10],d_lo[10]; uint32_t diff_zero,lo_one; int32_t c10_one; } PostArgs;
typedef struct { uint32_t width,height,w4,h4,n8w,n8h; int32_t m_fwd[9],m_inv[9]; int32_t thr_y,a_y,g20,g24,lo_l,hi_l,gate,g34,g38,g3c,g40; float f; int32_t inset;
    int32_t lum_on,bnr,blk,nbw,nbh; uint32_t d0; int32_t cc,c8,c4,c0,bc,b8,b4,b0,ac,a8,l98,l90,l88,t7c,t80; int32_t up_t[4]; } MarbleArgs;
typedef struct { MarbleArgs m; uint32_t shift; } MarblePackArgs;
// ZcTaskMarble on whole planes: the luma path when lum_on, the chroma path when bnr.
// tables: initial dec/enc, final dec/enc (0x4000 int16 each); upw: chroma upsample weights.
// post == NULL runs Marble alone. Otherwise the final pixel feeds Effect,
// ToneCurve and ShadeCorrection in the same kernel; fix/nfix receive the
// pixels that must be recomputed by the host reference implementation.
int sonylook_metal_marble(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,const MarbleArgs *args,const int16_t *dec_in,const int16_t *enc_in,const int16_t *dec_out,const int16_t *enc_out,const int32_t *upw,uint32_t upw_len,const PostArgs *post,const int16_t *pdec_in,const int16_t *penc_in,const int16_t *pdec_out,const int16_t *penc_out,const uint16_t *luts,uint16_t *ro,uint16_t *go,uint16_t *bo,uint32_t *fix,uint32_t *nfix,uint16_t *fixrgb,uint16_t *packed,uint32_t shift) { @autoreleasepool {
    if(!device||!r||!g||!b||w<10||h<10||!args||!dec_in||!enc_in||!dec_out||!enc_out||!upw)return fail(@"Invalid Metal Marble arguments");
    if(!packed&&(!ro||!go||!bo))return fail(@"Invalid Metal Marble output arguments");
    if(post&&(!pdec_in||!penc_in||!pdec_out||!penc_out||!luts||!fix||!nfix||!fixrgb))return fail(@"Invalid fused Marble + Post arguments");
    if(post&&batch_cmd)return fail(@"A fused Marble + Post pass must own its command buffer");
    size_t n=(size_t)w*h;MarbleArgs a=*args;a.width=w;a.height=h;
    a.w4=(w+3)&~3u;a.h4=(h+3)&~3u;a.n8w=(a.w4+7)>>3;a.n8h=(a.h4+7)>>3;
    if(a.bnr&&(a.blk!=4&&a.blk!=8))return fail(@"Marble chroma block size must be 4 or 8");
    a.nbw=(int32_t)((a.w4+a.blk-1)/(a.blk?a.blk:1));a.nbh=(int32_t)((a.h4+a.blk-1)/(a.blk?a.blk:1));
    size_t hn=(size_t)(a.w4/2)*a.h4;
    id rb=in_buf(r,n*2),gb=in_buf(g,n*2),bb=in_buf(b,n*2),ub=in_buf(upw,(size_t)upw_len*4);
    id t[4];const int16_t *tabs[4]={dec_in,enc_in,dec_out,enc_out};
    for(int i=0;i<4;i++){t[i]=in_buf(tabs[i],0x4000*2);if(!t[i])return fail(@"Metal Marble buffer allocation failed");}
    NSArray *ycc=new_buffers(3,n*2,MTLResourceStorageModePrivate),*half=new_buffers(2,hn*2,MTLResourceStorageModePrivate),*rgb=packed?nil:out3(ro,go,bo,n*2);
    id packed_buf=packed?out_buf(packed,n*6):nil;
    id<MTLBuffer> m=acquire((size_t)a.n8w*a.n8h*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> q=acquire((size_t)(a.n8w+12)*(a.n8h+12)*4,MTLResourceStorageModePrivate);
    id<MTLBuffer> q2=acquire((size_t)a.n8w*a.n8h*4,MTLResourceStorageModePrivate);
    if(!rb||!gb||!bb||!ub||!ycc||!half||(!packed&&!rgb)||(packed&&!packed_buf)||!m||!q||!q2)return fail(@"Metal Marble buffer allocation failed");
    PostArgs pa;const int16_t *ptabs[4]={pdec_in,penc_in,pdec_out,penc_out};id pb[4]={nil,nil,nil,nil},mab=nil,lb=nil,fb=nil,cb=nil,fib=nil;
    uint32_t zero=0;
    if(post) {
        pa=*post;pa.width=w;pa.height=h;
        mab=in_buf(&a,sizeof(a));
        for(int i=0;i<4;i++){pb[i]=in_buf(ptabs[i],0x4000*2);if(!pb[i])return fail(@"Metal fused Post table allocation failed");}
        lb=in_buf(luts,4*0x10000*2);
        fb=acquire((size_t)(pa.capacity?pa.capacity:1)*4,MTLResourceStorageModeShared);
        fib=acquire((size_t)(pa.capacity?pa.capacity:1)*6,MTLResourceStorageModeShared);
        cb=in_buf(&zero,4);
        if(!mab||!lb||!fb||!fib||!cb)return fail(@"Metal fused Post buffer allocation failed");
    }
    id<MTLCommandBuffer> cmd=command_buffer();
    if(getenv("SONYLOOK_MARBLE_HALFC")&&strcmp(getenv("SONYLOOK_MARBLE_HALFC"),"0")!=0){   // separate passes, for comparison
        encode_grid(cmd,27,[@[rb,gb,bb,t[0],t[1]] arrayByAddingObjectsFromArray:ycc],&a,sizeof(a),w,h);
        encode_grid(cmd,35,@[ycc[1],ycc[2],half[0],half[1]],&a,sizeof(a),a.w4/2,a.h4);
    } else encode_grid(cmd,55,[[@[rb,gb,bb,t[0],t[1]] arrayByAddingObjectsFromArray:ycc] arrayByAddingObjectsFromArray:half],&a,sizeof(a),a.w4/2,a.h4);
    if(a.lum_on) {
        encode_grid(cmd,28,@[ycc[0],m],&a,sizeof(a),a.n8w,a.n8h);
        encode_grid(cmd,29,@[m,q],&a,sizeof(a),a.n8w+4,a.n8h+4);
        encode_grid(cmd,30,@[q,q2],&a,sizeof(a),a.n8w,a.n8h);
    }
    NSArray *hc=half;
    if(a.bnr) {
        size_t nb=(size_t)a.nbw*a.nbh,np=(size_t)(a.nbw+12)*(a.nbh+12);
        NSArray *mb=new_buffers(3,nb*4,MTLResourceStorageModePrivate),*qp=new_buffers(2,np*4,MTLResourceStorageModePrivate);
        NSArray *qc=new_buffers(2,nb*4,MTLResourceStorageModePrivate),*oc=new_buffers(2,hn*2,MTLResourceStorageModePrivate);
        if(!mb||!qp||!qc||!oc)return fail(@"Metal Marble buffer allocation failed");
        encode_grid(cmd,36,[@[ycc[0],half[0],half[1]] arrayByAddingObjectsFromArray:mb],&a,sizeof(a),a.nbw,a.nbh);
        encode_grid(cmd,37,[mb arrayByAddingObjectsFromArray:qp],&a,sizeof(a),a.nbw+4,a.nbh+4);
        MarbleArgs ab=a;ab.n8w=a.nbw;ab.n8h=a.nbh;ab.a_y=a.ac;
        encode_grid(cmd,30,@[qp[0],qc[0]],&ab,sizeof(ab),a.nbw,a.nbh);
        MarbleArgs ar=ab;ar.a_y=a.a8;
        encode_grid(cmd,30,@[qp[1],qc[1]],&ar,sizeof(ar),a.nbw,a.nbh);
        encode_grid(cmd,38,@[qc[0],qc[1],ub,half[1],oc[0],oc[1]],&a,sizeof(a),a.w4/2,a.h4);
        hc=oc;
    }
    if(post) {
        // Packed: one interleaved output, bound in all three plane slots.
        NSArray *outs=packed?@[packed_buf,packed_buf,packed_buf]:rgb;
        NSArray *buffers=[[[[[ycc arrayByAddingObjectsFromArray:@[q2,rb,gb,bb,t[2],t[3]]] arrayByAddingObjectsFromArray:hc]
                           arrayByAddingObject:mab]
                           arrayByAddingObjectsFromArray:@[pb[0],pb[1],pb[2],pb[3],lb]]
                          arrayByAddingObjectsFromArray:outs];
        buffers=[buffers arrayByAddingObjectsFromArray:@[fb,cb,fib]];
        uint32_t layout[2]={packed?1u:0u,shift};
        id<MTLComputeCommandEncoder> e=new_encoder(cmd,46);[e setComputePipelineState:pipelines[46]];
        for(NSUInteger i=0;i<buffers.count;i++)bind(e,buffers[i],i);
        [e setBytes:&pa length:sizeof(pa) atIndex:buffers.count];
        [e setBytes:layout length:sizeof(layout) atIndex:buffers.count+1];
        [e dispatchThreads:MTLSizeMake(w,h,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];
    } else if(packed) {
        MarblePackArgs pa={a,shift};
        NSArray *buffers=[[ycc arrayByAddingObjectsFromArray:@[q2,rb,gb,bb,t[2],t[3]]] arrayByAddingObjectsFromArray:@[packed_buf]];
        buffers=[buffers arrayByAddingObjectsFromArray:hc];
        encode_grid(cmd,49,buffers,&pa,sizeof(pa),w,h);
    } else {
        encode_grid(cmd,31,[[[ycc arrayByAddingObjectsFromArray:@[q2,rb,gb,bb,t[2],t[3]]] arrayByAddingObjectsFromArray:rgb] arrayByAddingObjectsFromArray:hc],&a,sizeof(a),w,h);
    }
    int rc=submit_command(cmd);if(rc)return rc;
    if(packed)copy_back(packed_buf,packed,n*6);
    else {copy_back(rgb[0],ro,n*2);copy_back(rgb[1],go,n*2);copy_back(rgb[2],bo,n*2);}
    if(post){
        *nfix=*(uint32_t *)[cb contents];
        size_t kept=(size_t)MIN(*nfix,pa.capacity);
        memcpy(fix,[fb contents],kept*4);memcpy(fixrgb,[fib contents],kept*6);
    }
    return done(0);
}}

// Effect, ToneCurve and ShadeCorrection. Pixels the GPU cannot decide exactly
// are listed in fix[0..min(*nfix, capacity)); the caller recomputes them.
int sonylook_metal_post(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,const PostArgs *args,const int16_t *dec_in,const int16_t *enc_in,const int16_t *dec_out,const int16_t *enc_out,const uint16_t *luts,uint16_t *ro,uint16_t *go,uint16_t *bo,uint32_t *fix,uint32_t *nfix) { @autoreleasepool {
    if(!device||!r||!g||!b||!w||!h||!args||!dec_in||!enc_in||!dec_out||!enc_out||!luts||!ro||!go||!bo||!fix||!nfix)return fail(@"Invalid Metal post arguments");
    size_t n=(size_t)w*h;PostArgs a=*args;a.width=w;a.height=h;
    id<MTLBuffer> rb=in_buf(r,n*2);
    id<MTLBuffer> gb=in_buf(g,n*2);
    id<MTLBuffer> bb=in_buf(b,n*2);
    id<MTLBuffer> t[4];const int16_t *tabs[4]={dec_in,enc_in,dec_out,enc_out};
    for(int i=0;i<4;i++){t[i]=in_buf(tabs[i],0x4000*2);if(!t[i])return fail(@"Metal post buffer allocation failed");}
    id<MTLBuffer> lb=in_buf(luts,4*0x10000*2);
    id<MTLBuffer> fb=acquire((size_t)(a.capacity?a.capacity:1)*4,MTLResourceStorageModeShared);
    uint32_t zero=0;id<MTLBuffer> cb=in_buf(&zero,4);
    NSArray *rgb=out3(ro,go,bo,n*2);
    if(!rb||!gb||!bb||!lb||!fb||!cb||!rgb)return fail(@"Metal post buffer allocation failed");
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1;
    encode_grid(cmd,32,[[@[rb,gb,bb,t[0],t[1],t[2],t[3],lb] arrayByAddingObjectsFromArray:rgb] arrayByAddingObjectsFromArray:@[fb,cb]],&a,sizeof(a),w,h);
    [cmd commit];[cmd waitUntilCompleted];if(cmd.error)return fail(cmd.error.description);
    copy_back(rgb[0],ro,n*2);copy_back(rgb[1],go,n*2);copy_back(rgb[2],bo,n*2);
    *nfix=*(uint32_t *)[cb contents];memcpy(fix,[fb contents],(size_t)MIN(*nfix,a.capacity)*4);return done(0);
}}

// ZcTaskRawNR on the whole raw. thr: three 0x8000-entry threshold tables.
int sonylook_metal_rawnr(const uint16_t *raw,uint32_t w,uint32_t h,const NrArgs *args,const int32_t *thr,uint16_t *out) { @autoreleasepool {
    if(!device||!raw||!out||!args||!thr||w<21||h<21)return fail(@"Invalid Metal RawNR arguments");
    size_t n=(size_t)w*h;NrArgs a=*args;a.width=w;a.height=h;
    id rb=in_buf(raw,n*2),tb=in_buf(thr,3*0x8000*4),ob=out_buf(out,n*2);
    id<MTLBuffer> hp=acquire(n*2,MTLResourceStorageModePrivate),base=acquire(n*2,MTLResourceStorageModePrivate);
    if(!rb||!tb||!ob||!hp||!base)return fail(@"Metal RawNR buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();
    encode_band(cmd,33,@[rb,hp,base],&a,sizeof(a),w,0,h);
    encode_band(cmd,34,@[rb,hp,base,tb,ob],&a,sizeof(a),w,0,h);
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(ob,out,n*2);return done(0);
}}

// data[i] = lut[data[i]] in place on a GPU-resident uint16 array (an output
// re-encode, e.g. ProPhoto's gamma 1.8). Runs on its own; not inside a batch.
int sonylook_metal_lut_u16(uint16_t *data,size_t n,const uint16_t *lut,uint32_t lut_len) { @autoreleasepool {
    if(!device||!data||!lut||!n||!lut_len||n>0xffffffffu)return fail(@"Invalid Metal LUT arguments");
    id db=find_resident(data,n*2);
    if(!db)return fail(@"Metal LUT needs a GPU-resident array");
    id lb=in_buf(lut,(size_t)lut_len*2);
    if(!lb)return fail(@"Metal LUT allocation failed");
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1;
    uint32_t args[2]={(uint32_t)n,lut_len};
    id<MTLComputeCommandEncoder> e=new_encoder(cmd,57);[e setComputePipelineState:pipelines[57]];
    bind(e,db,0);bind(e,lb,1);[e setBytes:args length:sizeof(args) atIndex:2];
    [e dispatchThreads:MTLSizeMake((n+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
    int rc=submit_command(cmd);return done(rc);
}}
// Three uint16 planes -> one H x W x 3 interleaved image. shift=2 writes
// Edit's 16-bit TIFF values directly (14-bit encoded value << 2).
int sonylook_metal_interleave(const uint16_t *r,const uint16_t *g,const uint16_t *b,uint32_t w,uint32_t h,uint32_t shift,uint16_t *out) { @autoreleasepool {
    if(!device||!r||!g||!b||!out||!w||!h)return fail(@"Invalid Metal interleave arguments");
    size_t n=(size_t)w*h;struct {uint32_t width,height,shift;} dims={w,h,shift};
    id rb=in_buf(r,n*2),gb=in_buf(g,n*2),bb=in_buf(b,n*2),ob=out_buf(out,n*6);
    if(!rb||!gb||!bb||!ob)return fail(@"Metal interleave buffer allocation failed");
    id<MTLCommandBuffer> cmd=command_buffer();
    encode_grid(cmd,40,@[rb,gb,bb,ob],&dims,sizeof(dims),w,h);
    int rc=submit_command(cmd);if(rc)return rc;
    copy_back(ob,out,n*6);return done(0);
}}

// Test hook: DRO's exp2 and division on n inputs.
int sonylook_metal_dro_math(const float *a,const float *b,uint32_t n,float *e2,float *q) { @autoreleasepool {
    if(!device||!a||!b||!e2||!q||!n)return fail(@"Invalid Metal DRO math arguments");
    id ab=in_buf(a,(size_t)n*4),bb=in_buf(b,(size_t)n*4),eb=out_buf(e2,(size_t)n*4),qb=out_buf(q,(size_t)n*4);
    if(!ab||!bb||!eb||!qb)return fail(@"Metal DRO math buffer allocation failed");
    id<MTLCommandBuffer> cmd=own_command_buffer();if(!cmd)return -1;id<MTLComputeCommandEncoder> e=new_encoder(cmd,39);
    [e setComputePipelineState:pipelines[39]];NSArray *bufs=@[ab,bb,eb,qb];
    for(NSUInteger i=0;i<bufs.count;i++)bind(e,bufs[i],i);
    [e setBytes:&n length:4 atIndex:4];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
    [cmd commit];[cmd waitUntilCompleted];if(cmd.error)return fail(cmd.error.description);
    copy_back(eb,e2,(size_t)n*4);copy_back(qb,q,(size_t)n*4);return done(0);
}}
