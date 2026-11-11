// HEIF (.hif / .heic) output through Core Image, which encodes HEVC on the
// hardware encoder. Built into libsonylook_metal.dylib next to bridge.m (kept
// apart: the Core Image headers clash with bridge.m's static names).
#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#include <stdint.h>
#include <stdio.h>

static _Thread_local char heif_error[1024];
const char *sonylook_heif_error(void) { return heif_error; }
static int heif_fail(NSString *s) { snprintf(heif_error,sizeof(heif_error),"%s",s.UTF8String); return -1; }

// rgb: interleaved uint16 RGB on the 16-bit scale (the 16-bit TIFF values,
// 14-bit << 2); icc: the output profile, embedded in the file. depth 10
// writes a 10-bit HEIF, 8 an 8-bit one; quality is 0..1. Thread-safe.
int sonylook_heif_write(const uint16_t *rgb,uint32_t w,uint32_t h,const void *icc,size_t icc_len,
                        float quality,int depth,const char *path) { @autoreleasepool {
    if(!rgb||!w||!h||!icc||!icc_len||!path||(depth!=8&&depth!=10))return heif_fail(@"Invalid HEIF arguments");
    static CIContext *ctx;static dispatch_once_t once;
    dispatch_once(&once,^{ctx=[CIContext contextWithOptions:@{kCIContextCacheIntermediates:@NO}];});
    CGColorSpaceRef cs=CGColorSpaceCreateWithICCData((__bridge CFDataRef)[NSData dataWithBytes:icc length:icc_len]);
    if(!cs)return heif_fail(@"HEIF: the ICC profile is not usable");
    NSData *data=[NSData dataWithBytesNoCopy:(void *)rgb length:(size_t)w*h*6 freeWhenDone:NO];
    CGDataProviderRef dp=CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef cg=CGImageCreate(w,h,16,48,(size_t)w*6,cs,kCGBitmapByteOrder16Little|(CGBitmapInfo)kCGImageAlphaNone,
                                dp,NULL,false,kCGRenderingIntentDefault);
    CGDataProviderRelease(dp);
    if(!cg){CGColorSpaceRelease(cs);return heif_fail(@"HEIF: cannot wrap the image");}
    CIImage *im=[CIImage imageWithCGImage:cg];
    NSDictionary *opts=@{(id)kCGImageDestinationLossyCompressionQuality:@(quality)};
    NSURL *url=[NSURL fileURLWithPath:@(path)];
    NSError *err=nil;BOOL ok;
    if(depth==10) ok=[ctx writeHEIF10RepresentationOfImage:im toURL:url colorSpace:cs options:opts error:&err];
    else ok=[ctx writeHEIFRepresentationOfImage:im toURL:url format:kCIFormatRGBA8 colorSpace:cs options:opts error:&err];
    CGImageRelease(cg);CGColorSpaceRelease(cs);
    return ok?0:heif_fail(err?[@"HEIF write failed: " stringByAppendingString:err.localizedDescription]:@"HEIF write failed");
}}

// For checks: a HEIF's size, bit depth and profile name (info), and its pixels
// as RGB16 in the file's own color space (read; out holds w*h*3 values).
int sonylook_heif_info(const char *path,uint32_t *w,uint32_t *h,uint32_t *depth,char *profile,size_t profile_len) { @autoreleasepool {
    CGImageSourceRef src=CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:@(path)],NULL);
    if(!src)return heif_fail(@"Cannot open the HEIF file");
    NSDictionary *p=(__bridge_transfer NSDictionary *)CGImageSourceCopyPropertiesAtIndex(src,0,NULL);
    CFRelease(src);
    if(!p)return heif_fail(@"Cannot read the HEIF properties");
    *w=[p[(id)kCGImagePropertyPixelWidth] unsignedIntValue];*h=[p[(id)kCGImagePropertyPixelHeight] unsignedIntValue];
    *depth=[p[(id)kCGImagePropertyDepth] unsignedIntValue];
    NSString *name=p[(id)kCGImagePropertyProfileName]?:@"";
    snprintf(profile,profile_len,"%s",name.UTF8String);
    return 0;
}}
int sonylook_heif_read(const char *path,uint16_t *out,uint32_t w,uint32_t h) { @autoreleasepool {
    CGImageSourceRef src=CGImageSourceCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:@(path)],NULL);
    if(!src)return heif_fail(@"Cannot open the HEIF file");
    CGImageRef img=CGImageSourceCreateImageAtIndex(src,0,NULL);CFRelease(src);
    if(!img||CGImageGetWidth(img)!=w||CGImageGetHeight(img)!=h){if(img)CGImageRelease(img);return heif_fail(@"HEIF size mismatch");}
    uint16_t *rgbx=malloc((size_t)w*h*8);
    CGContextRef c=CGBitmapContextCreate(rgbx,w,h,16,(size_t)w*8,CGImageGetColorSpace(img),
                                         kCGBitmapByteOrder16Little|(CGBitmapInfo)kCGImageAlphaNoneSkipLast);
    if(!c){free(rgbx);CGImageRelease(img);return heif_fail(@"Cannot create a bitmap context");}
    CGContextDrawImage(c,CGRectMake(0,0,w,h),img);
    CGContextRelease(c);CGImageRelease(img);
    for(size_t i=0;i<(size_t)w*h;i++){out[3*i]=rgbx[4*i];out[3*i+1]=rgbx[4*i+1];out[3*i+2]=rgbx[4*i+2];}
    free(rgbx);
    return 0;
}}
