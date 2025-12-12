# python/CMakeLists.txt — Build rules for the Python extension module.

cmake_minimum_required(VERSION 3.16)

include(FetchContent)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.1
)
FetchContent_MakeAvailable(pybind11)

set(T81LIB_PYTHON_MODULE_NAME "t81lib")

if(TORCH_CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
endif()

pybind11_add_module(t81lib_python MODULE bindings.cpp)

target_sources(t81lib_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/t81/core/gguf_quants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/linalg/gemm_dispatch.cpp)

target_compile_features(t81lib_python PRIVATE cxx_std_20)

#target_link_libraries(t81lib_python PRIVATE t81lib)

target_compile_definitions(
    t81lib_python
    PRIVATE T81LIB_TORCH_ENABLED=$<BOOL:${T81LIB_ENABLE_TORCH_BINDINGS}>)

target_include_directories(t81lib_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_definitions(t81lib_python PRIVATE
    T81LIB_USE_CUDA=$<BOOL:${USE_CUDA}>
    T81LIB_USE_ROCM=$<BOOL:${USE_ROCM}>)

if(T81LIB_ENABLE_TORCH_BINDINGS)
    find_package(Torch CONFIG REQUIRED)
    target_link_libraries(t81lib_python PRIVATE Torch::Torch)
endif()

if(USE_CUDA)
    target_sources(t81lib_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/linalg/gemm_cuda.cu)
    set_target_properties(t81lib_python PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(t81lib_python PRIVATE CUDA::cudart)
endif()

if(USE_ROCM)
    target_sources(t81lib_python PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/linalg/gemm_rocm.cpp)
endif()

set_target_properties(t81lib_python PROPERTIES
    OUTPUT_NAME ${T81LIB_PYTHON_MODULE_NAME}
    PREFIX ""
)
