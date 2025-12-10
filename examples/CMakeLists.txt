add_executable(example_basic example_basic.cpp)
target_link_libraries(example_basic PRIVATE t81::t81lib)

add_executable(example_montgomery example_montgomery.cpp)
target_link_libraries(example_montgomery PRIVATE t81::t81lib)
