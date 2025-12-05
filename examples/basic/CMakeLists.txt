cmake_minimum_required(VERSION 3.20)
project(t81lib_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(example_basic main.cpp)
target_link_libraries(example_basic PRIVATE t81lib)
target_compile_features(example_basic PRIVATE cxx_std_20)
