# tests/CMakeLists.txt — Configures the unit test targets.

add_executable(test_limb_basic
    unit/test_limb_basic.cpp
)
target_link_libraries(test_limb_basic PRIVATE t81::t81lib)
add_test(NAME limb_basic COMMAND test_limb_basic)

add_executable(test_bigint_basic
    unit/test_bigint_basic.cpp
)
target_link_libraries(test_bigint_basic PRIVATE t81::t81lib)
add_test(NAME bigint_basic COMMAND test_bigint_basic)

add_executable(test_bigint_ops
    unit/test_bigint_ops.cpp
)
target_link_libraries(test_bigint_ops PRIVATE t81::t81lib)
add_test(NAME bigint_ops COMMAND test_bigint_ops)

add_executable(test_bigint_bitops_fuzz
    unit/test_bigint_bitops_fuzz.cpp
)
target_link_libraries(test_bigint_bitops_fuzz PRIVATE t81::t81lib)
add_test(NAME bigint_bitops_fuzz COMMAND test_bigint_bitops_fuzz)

add_executable(test_montgomery
    unit/test_montgomery.cpp
)
target_link_libraries(test_montgomery PRIVATE t81::t81lib)
add_test(NAME montgomery COMMAND test_montgomery)

add_executable(test_limb_float_hash
    unit/test_limb_float_hash.cpp
)
target_link_libraries(test_limb_float_hash PRIVATE t81::t81lib)
add_test(NAME limb_float_hash COMMAND test_limb_float_hash)

add_executable(test_limb_properties
    unit/test_limb_properties.cpp
)
target_link_libraries(test_limb_properties PRIVATE t81::t81lib)
add_test(NAME limb_properties COMMAND test_limb_properties)
