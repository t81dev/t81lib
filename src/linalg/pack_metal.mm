#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <Metal/Metal.h>

#include <stdexcept>

#include <t81/core/limb.hpp>
#include <t81/linalg/pack_gpu.hpp>

#if !defined(PACK_METAL_LIBRARY_PATH)
#error "PACK_METAL_LIBRARY_PATH must be defined when compiling the Metal pack backend"
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
    NSString *path = @(PACK_METAL_LIBRARY_PATH);
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithFile:path error:&error];
    if (!library) {
        throw std::runtime_error([[error localizedDescription] UTF8String]);
    }
    return library;
}

id<MTLComputePipelineState> quantize_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        id<MTLLibrary> library = metal_library(device);
        NSError *error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"quantize_trits_kernel"];
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            throw std::runtime_error([[error localizedDescription] UTF8String]);
        }
    });
    return pipeline;
}

id<MTLComputePipelineState> pack_pipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        id<MTLLibrary> library = metal_library(device);
        NSError *error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@"pack_dense_matrix_kernel"];
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            throw std::runtime_error([[error localizedDescription] UTF8String]);
        }
    });
    return pipeline;
}

} // namespace

void metal_quantize_to_trits(std::span<const float> src,
                             std::span<std::int8_t> dst,
                             float threshold) {
    if (src.size() != dst.size()) {
        throw std::invalid_argument("quantize_to_trits size mismatch");
    }
    if (src.empty()) {
        return;
    }
    id<MTLDevice> device = metal_device();
    if (!device) {
        throw std::runtime_error("Metal device not available");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = quantize_pipeline(device);

    const std::size_t src_size = src.size() * sizeof(float);
    const std::size_t dst_size = dst.size() * sizeof(std::int8_t);

    id<MTLBuffer> src_buffer =
        [device newBufferWithBytes:src.data() length:src_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buffer = [device newBufferWithBytesNoCopy:dst.data()
                                                         length:dst_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];

    struct QuantParams {
        std::uint32_t count;
        float threshold;
    } params{static_cast<std::uint32_t>(src.size()), threshold};

    id<MTLBuffer> params_buffer =
        [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buffer offset:0 atIndex:0];
    [encoder setBuffer:dst_buffer offset:0 atIndex:1];
    [encoder setBuffer:params_buffer offset:0 atIndex:2];

    const MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    const MTLSize gridSize = MTLSizeMake(params.count, 1, 1);
    const MTLSize groupCount = MTLSizeMake(
        (gridSize.width + threadsPerGroup.width - 1) / threadsPerGroup.width, 1, 1);

    [encoder dispatchThreadgroups:groupCount threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];

    [queue release];
    [src_buffer release];
    [dst_buffer release];
    [params_buffer release];
}

void metal_pack_dense_matrix(std::span<const float> src,
                             std::span<std::uint8_t> dst,
                             int rows,
                             int cols,
                             float threshold) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("pack_dense_matrix dimensions must be non-negative");
    }
    const int trits_per_limb = core::limb::TRITS;
    const int limbs_per_row = (cols + trits_per_limb - 1) / trits_per_limb;
    const std::size_t limb_bytes = static_cast<std::size_t>(core::limb::BYTES);
    const std::size_t expected_src =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t expected_dst = static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(limbs_per_row) * limb_bytes;
    if (src.size() != expected_src || dst.size() != expected_dst) {
        throw std::invalid_argument("pack_dense_matrix buffer sizes do not match");
    }
    if (src.empty()) {
        return;
    }

    id<MTLDevice> device = metal_device();
    if (!device) {
        throw std::runtime_error("Metal device not available");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    id<MTLComputePipelineState> pipeline = pack_pipeline(device);

    const std::size_t src_size = src.size() * sizeof(float);
    const std::size_t dst_size = dst.size() * sizeof(std::uint8_t);

    id<MTLBuffer> src_buffer =
        [device newBufferWithBytes:src.data() length:src_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst_buffer = [device newBufferWithBytesNoCopy:dst.data()
                                                         length:dst_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];

    struct PackParams {
        std::uint32_t rows;
        std::uint32_t cols;
        std::uint32_t limbs_per_row;
        std::uint32_t trits_per_limb;
        std::uint32_t limb_bytes;
        float threshold;
    } params{
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(cols),
        static_cast<std::uint32_t>(limbs_per_row),
        static_cast<std::uint32_t>(trits_per_limb),
        static_cast<std::uint32_t>(limb_bytes),
        threshold,
    };

    id<MTLBuffer> params_buffer =
        [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src_buffer offset:0 atIndex:0];
    [encoder setBuffer:dst_buffer offset:0 atIndex:1];
    [encoder setBuffer:params_buffer offset:0 atIndex:2];

    const std::uint32_t total_limbs = params.rows * params.limbs_per_row;
    const MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    const MTLSize gridSize = MTLSizeMake(total_limbs, 1, 1);
    const MTLSize groupCount = MTLSizeMake(
        (gridSize.width + threadsPerGroup.width - 1) / threadsPerGroup.width, 1, 1);

    [encoder dispatchThreadgroups:groupCount threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];

    [queue release];
    [src_buffer release];
    [dst_buffer release];
    [params_buffer release];
}

#endif

} // namespace t81::linalg::detail
