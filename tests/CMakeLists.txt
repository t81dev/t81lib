add_library(t81lib_test_main OBJECT
    unit/test_main.cpp
)

target_link_libraries(t81lib_test_main
    PUBLIC
        t81::t81lib
)

add_executable(test_limb_basic
    unit/test_limb_basic.cpp
    $<TARGET_OBJECTS:t81lib_test_main>
)
target_link_libraries(test_limb_basic PRIVATE t81::t81lib)
add_test(NAME limb_basic COMMAND test_limb_basic)

add_executable(test_bigint_basic
    unit/test_bigint_basic.cpp
    $<TARGET_OBJECTS:t81lib_test_main>
)
target_link_libraries(test_bigint_basic PRIVATE t81::t81lib)
add_test(NAME bigint_basic COMMAND test_bigint_basic)
