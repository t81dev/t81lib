cmake_minimum_required(VERSION 3.20)
project(t81lib-bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force high optimization and native code paths for benchmarking.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")

include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Benchmark tests" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(t81lib-bench
    main.cpp
    addition.cpp
    number_theory.cpp
    multiplication.cpp
    negation.cpp
    subtraction.cpp
    addc_chain.cpp
    square.cpp
    multi_limb.cpp
    multi_limb_large.cpp
    gmp_comparison.cpp
    conversion.cpp
    shift.cpp
    pow_mod.cpp
    compare.cpp
    binary_gods.cpp
    competitors/gmp.cpp
    competitors/boost_multiprecision.cpp
    competitors/ttmath.cpp
    competitors/naive_ternary.cpp
    division.cpp
)

target_link_libraries(t81lib-bench PRIVATE
    t81lib
    benchmark::benchmark
    benchmark::benchmark_main
)

set(GMP_DEPS_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps")
if(EXISTS "${GMP_DEPS_DIR}")
    target_include_directories(t81lib-bench PRIVATE "${GMP_DEPS_DIR}")
endif()
if(EXISTS "${GMP_DEPS_DIR}/include")
    target_include_directories(t81lib-bench PRIVATE "${GMP_DEPS_DIR}/include")
endif()
if(EXISTS "${GMP_DEPS_DIR}/lib")
    target_link_directories(t81lib-bench PRIVATE "${GMP_DEPS_DIR}/lib")
    target_link_libraries(t81lib-bench PRIVATE gmp gmpxx)
endif()

if(EXISTS "${GMP_DEPS_DIR}/gmp.h")
    target_compile_definitions(t81lib-bench PRIVATE T81BENCH_GMP_ENABLED=1)
endif()
if(EXISTS "${GMP_DEPS_DIR}/boost/multiprecision/cpp_int.hpp")
    target_compile_definitions(t81lib-bench PRIVATE T81BENCH_BOOST_ENABLED=1)
endif()
if(EXISTS "${GMP_DEPS_DIR}/ttmath.h")
    file(READ "${GMP_DEPS_DIR}/ttmath.h" TTMATH_CONTENT)
    if(NOT TTMATH_CONTENT MATCHES "404: Not Found")
        target_compile_definitions(t81lib-bench PRIVATE T81BENCH_TTMATH_ENABLED=1)
    endif()
elseif(EXISTS "${GMP_DEPS_DIR}/ttmath/ttmath.h")
    file(READ "${GMP_DEPS_DIR}/ttmath/ttmath.h" TTMATH_CONTENT)
    if(NOT TTMATH_CONTENT MATCHES "404: Not Found")
        target_compile_definitions(t81lib-bench PRIVATE T81BENCH_TTMATH_ENABLED=1)
    endif()
endif()

target_compile_features(t81lib-bench PRIVATE cxx_std_20)


file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)
