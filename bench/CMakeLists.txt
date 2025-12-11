# bench/CMakeLists.txt — CMake targets for running the benchmarks.

add_executable(bench_limb_add bench_limb_add.cpp)
target_link_libraries(bench_limb_add PRIVATE benchmark::benchmark t81::t81lib)

add_executable(bench_bigint_montgomery bench_bigint_montgomery.cpp)
target_link_libraries(bench_bigint_montgomery PRIVATE benchmark::benchmark t81::t81lib)

add_executable(bench_limb_ops bench_limb_ops.cpp)
target_link_libraries(bench_limb_ops PRIVATE benchmark::benchmark t81::t81lib)

add_executable(t81-bench bench_t81.cpp)
target_link_libraries(t81-bench PRIVATE benchmark::benchmark t81::t81lib)

add_executable(bench_numeric_types bench_numeric_types.cpp)
target_link_libraries(bench_numeric_types PRIVATE benchmark::benchmark t81::t81lib)
