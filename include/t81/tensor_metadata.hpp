#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace t81 {

enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    INVALID,
};

enum class ScalarType {
    Byte,
    Char,
    Short,
    Int,
    Long,
    Half,
    Float,
    Double,
    BFloat16,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    Undefined,
};

struct TensorMetadata {
    DeviceType device_type = DeviceType::CPU;
    int32_t device_index = 0;
    void *data_ptr = nullptr;
    ScalarType dtype = ScalarType::Undefined;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    bool owns_memory = false;
    std::shared_ptr<void> storage_handle;
    int64_t storage_offset = 0;
    bool requires_sync = false;

    std::size_t numel() const {
        std::size_t value = 1;
        for (const int64_t size : sizes) {
            value *= static_cast<std::size_t>(size);
        }
        return value;
    }

    bool is_contiguous() const {
        if (sizes.empty()) {
            return true;
        }
        if (!strides.empty() && strides.size() != sizes.size()) {
            return false;
        }
        if (strides.empty()) {
            return true;
        }
        std::size_t expected = 1;
        for (int index = static_cast<int>(sizes.size()) - 1; index >= 0; --index) {
            if (strides[static_cast<std::size_t>(index)] != static_cast<int64_t>(expected)) {
                return false;
            }
            expected *= static_cast<std::size_t>(sizes[static_cast<std::size_t>(index)]);
        }
        return true;
    }

    bool dtype_is_float32() const {
        return dtype == ScalarType::Float;
    }
};

inline TensorMetadata make_contiguous_metadata(std::size_t elements, void *pointer) {
    TensorMetadata metadata;
    metadata.sizes = {static_cast<int64_t>(elements)};
    metadata.strides = {1};
    metadata.data_ptr = pointer;
    metadata.dtype = ScalarType::Float;
    return metadata;
}

} // namespace t81
