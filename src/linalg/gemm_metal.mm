#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <Metal/Metal.h>
#include <t81/core/limb.hpp>

#if !defined(GEMM_METAL_LIBRARY_PATH)
#error "GEMM_METAL_LIBRARY_PATH must be defined when compiling the Metal backend"
#endif

namespace t81::linalg::detail {

#if T81LIB_USE_METAL

namespace {

id<MTLDevice> metal_device() {
    static dispatch_once_t onceToken;
    static id<MTLDevice> device = nil;
    dispatch_once(&onceToken, ^{
        device = MTLCreateSystemDefaultDevice();
    });
    return device;
}

id<MTLLibrary> metal_library(id<MTLDevice> device) {
    NSString *path = @(GEMM_METAL_LIBRARY_PATH);
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
    if (!library) {
        throw std::runtime_error([[error localizedDescription] UTF8String]);
    }
    return library;
}

id<MTLComputePipelineState> metal_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        id<MTLLibrary> library = metal_library(device);
        NSError *error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"gemm_ternary_kernel"];
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            throw std::runtime_error([[error localizedDescription] UTF8String]);
        }
    });
    return pipeline;
}

} // namespace

void metal_gemm_ternary(std::span<const core::limb> A,
                        std::span<const core::limb> B,
                        std::span<float> C,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        float beta) {
    id<MTLDevice> device = metal_device();
    if (!device) {
        throw std::runtime_error("Metal device not available");
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];

    id<MTLComputePipelineState> pipeline = metal_pipeline(device);

    auto *a_ptr = reinterpret_cast<const uint8_t *>(A.data());
    auto *b_ptr = reinterpret_cast<const uint8_t *>(B.data());
    auto *c_ptr = C.data();
    const std::size_t a_size = A.size() * sizeof(core::limb);
    const std::size_t b_size = B.size() * sizeof(core::limb);
    const std::size_t c_size = C.size() * sizeof(float);

    id<MTLBuffer> a_buffer = [device newBufferWithBytes:a_ptr length:a_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buffer = [device newBufferWithBytes:b_ptr length:b_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_buffer = [device newBufferWithBytesNoCopy:c_ptr length:c_size options:MTLResourceStorageModeShared deallocator:nil];

    struct GemmParams {
        int M;
        int N;
        int K_limbs;
        float alpha;
        float beta;
    } params{M, N, K / core::limb::TRITS, alpha, beta};

    id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:b_buffer offset:0 atIndex:1];
    [encoder setBuffer:c_buffer offset:0 atIndex:2];
    [encoder setBuffer:params_buffer offset:0 atIndex:3];

    const MTLSize threadsPerGroup = MTLSizeMake(16, 1, 1);
    const MTLSize gridSize = MTLSizeMake(M * N, 1, 1);
    const MTLSize groupCount = MTLSizeMake(
        (gridSize.width + threadsPerGroup.width - 1) / threadsPerGroup.width,
        1,
        1);

    [encoder dispatchThreadgroups:groupCount threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];

    [queue release];
    [a_buffer release];
    [b_buffer release];
    [c_buffer release];
    [params_buffer release];
}

#endif

} // namespace t81::linalg::detail
