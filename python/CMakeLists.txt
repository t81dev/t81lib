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

pybind11_add_module(t81lib_python MODULE t81lib_module.cpp)

target_link_libraries(t81lib_python PRIVATE t81lib)

set_target_properties(t81lib_python PROPERTIES
    OUTPUT_NAME ${T81LIB_PYTHON_MODULE_NAME}
    PREFIX ""
)
