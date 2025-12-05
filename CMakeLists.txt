cmake_minimum_required(VERSION 3.20)
project(t81lib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(t81lib INTERFACE)
target_include_directories(t81lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/t81lib-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/t81lib-config.cmake
    INSTALL_DESTINATION lib/cmake/t81lib
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/t81lib-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

add_executable(basic_test tests/basic.cpp)
target_link_libraries(basic_test PRIVATE t81lib)
target_compile_features(basic_test PRIVATE cxx_std_20)

add_executable(unit_tests tests/unit_tests.cpp)
target_link_libraries(unit_tests PRIVATE t81lib)
target_compile_features(unit_tests PRIVATE cxx_std_20)

add_executable(fuzz_test tests/fuzz.cpp)
target_link_libraries(fuzz_test PRIVATE t81lib)
target_compile_features(fuzz_test PRIVATE cxx_std_20)
option(BUILD_BENCHMARKS "Enable building the benchmark suite" OFF)
option(BUILD_EXAMPLES "Build usage examples" ON)
option(ENABLE_SANITIZERS "Build the tests with address/undefined sanitizers" OFF)

if (BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if (BUILD_BENCHMARKS)
    set(T81LIB_HAS_BENCHMARKS 1)
else()
    set(T81LIB_HAS_BENCHMARKS 0)
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/t81)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/t81/t81lib_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/t81/t81lib_config.hpp
    @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/t81lib.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/t81lib.pc
    @ONLY
)

include(CTest)
enable_testing()
add_test(NAME basic COMMAND basic_test)
add_test(NAME unit_tests COMMAND unit_tests)
add_test(NAME fuzz COMMAND fuzz_test)
if (BUILD_EXAMPLES)
    add_subdirectory(examples/basic)
endif()

if (ENABLE_SANITIZERS)
    foreach(target basic_test unit_tests fuzz_test)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()

install(TARGETS t81lib EXPORT t81libTargets)
install(DIRECTORY include/ DESTINATION include)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/t81/t81lib_config.hpp
    DESTINATION include/t81
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/t81lib.pc
    DESTINATION lib/pkgconfig
)
install(EXPORT t81libTargets
    FILE t81libTargets.cmake
    NAMESPACE t81::
    DESTINATION lib/cmake/t81lib
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/t81lib-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/t81lib-config-version.cmake
    DESTINATION lib/cmake/t81lib
)
