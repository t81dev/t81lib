# GPU backends & tensor metadata

CUDA/ROCm kernels can be built when you configure with `-DUSE_CUDA=ON` or `-DUSE_ROCM=ON` (see `python/CMakeLists.txt`). The bindings expose `t81lib.where`, `t81lib.clamp`, `t81lib.lerp`, and `t81lib.addcmul`, which accept either NumPy buffers or PyTorch tensors and dispatch directly to the GPU kernels.

Dispatch relies on `t81::TensorMetadata` (`include/t81/tensor_metadata.hpp`): a lightweight struct that carries device tags, dtype codes, shape, strides, and `data_ptr` so the dispatcher can call the right CUDA/HIP kernel without copies. When torch is available, `t81lib` automatically wraps tensors; without torch it gracefully falls back to CPU buffers. Review `python/bindings.cpp` for the extraction helpers and lifetime management.
