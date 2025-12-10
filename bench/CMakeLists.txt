add_executable(bench_limb_add bench_limb_add.cpp)
target_link_libraries(bench_limb_add PRIVATE t81::t81lib)

add_executable(bench_bigint_montgomery bench_bigint_montgomery.cpp)
target_link_libraries(bench_bigint_montgomery PRIVATE t81::t81lib)
