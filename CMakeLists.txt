#CMakeLists.txt — Top - level CMake build configuration for the library.

cmake_minimum_required(VERSION 3.16) project(t81lib VERSION 0.1.0 LANGUAGES CXX)

    option(T81LIB_BUILD_TESTS "Build t81lib tests" ON) option(
        T81LIB_BUILD_BENCH "Build t81lib benchmarks" OFF) option(T81LIB_USE_GOOGLE_BENCH
                                                                 "Fetch Google Benchmark when a "
                                                                 "system package is missing" ON)
        option(T81LIB_HEADER_ONLY "Build t81lib as header-only interface library" ON) option(
            T81LIB_BUILD_PYTHON_BINDINGS
            "Build Python bindings for t81lib" ON) option(T81LIB_ENABLE_TORCH_BINDINGS
                                                          "Enable PyTorch-aware C++ helpers when "
                                                          "the [torch] extra is requested" OFF)

            set(TORCH_CMAKE_PREFIX_PATH
                "" CACHE STRING
                "Additional CMake prefix paths supplied by torch.utils.cmake_prefix_path")

                set(CMAKE_CXX_STANDARD 20) set(CMAKE_CXX_STANDARD_REQUIRED ON) set(
                    CMAKE_CXX_EXTENSIONS OFF)

                    add_library(t81lib INTERFACE) target_include_directories(
                        t81lib INTERFACE $<BUILD_INTERFACE : ${CMAKE_CURRENT_SOURCE_DIR} / include>
                            $<INSTALL_INTERFACE : include>)

                        target_compile_features(t81lib INTERFACE cxx_std_20)

                            include(CheckCXXCompilerFlag)

                                check_cxx_compiler_flag("-mavx512"
                                                        "f" T81LIB_SUPPORTS_GNU_AVX512F) if (
                                    T81LIB_SUPPORTS_GNU_AVX512F) target_compile_options(t81lib
                                                                                            INTERFACE -
                                                                                        mavx512f)
                                    endif() if (MSVC) check_cxx_compiler_flag("/arch:"
                                                                              "AVX51"
                                                                              "2" T81LIB_SUPPORTS_MSVC_AVX512) if (
                                        T81LIB_SUPPORTS_MSVC_AVX512) target_compile_options(t81lib
                                                                                                INTERFACE /
                                                                                            arch : AVX512) endif() endif()

#ARM targets default to enabling the NEON path.
                                        if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv8|"
                                                                           "armv7|armhf|armeabi)") target_compile_definitions(
                                            t81lib INTERFACE T81_ENABLE_NEON = 1) endif()

#Exported namespace : t81::t81lib
                                            add_library(t81::t81lib ALIAS t81lib) add_library(
                                                t81::t81 ALIAS t81lib)

                                                find_package(benchmark QUIET)

                                                    if (T81LIB_BUILD_BENCH AND NOT benchmark_FOUND AND NOT
                                                            T81LIB_USE_GOOGLE_BENCH) message(FATAL_ERROR
                                                                                             "t81li"
                                                                                             "b "
                                                                                             "bench"
                                                                                             "marks"
                                                                                             " requ"
                                                                                             "ire "
                                                                                             "Googl"
                                                                                             "e "
                                                                                             "Bench"
                                                                                             "mark;"
                                                                                             " inst"
                                                                                             "all "
                                                                                             "it "
                                                                                             "or "
                                                                                             "enabl"
                                                                                             "e "
                                                                                             "T81LI"
                                                                                             "B_"
                                                                                             "USE_"
                                                                                             "GOOGL"
                                                                                             "E_"
                                                                                             "BENCH"
                                                                                             ".") endif()

                                                        include(FetchContent) if (
                                                            T81LIB_BUILD_BENCH AND NOT benchmark_FOUND AND
                                                                T81LIB_USE_GOOGLE_BENCH) set(BENCHMARK_ENABLE_TESTING
                                                                                                 OFF CACHE BOOL
                                                                                             "Disab"
                                                                                             "le "
                                                                                             "googl"
                                                                                             "etest"
                                                                                             " disc"
                                                                                             "over"
                                                                                             "y" FORCE)
                                                            FetchContent_Declare(
                                                                googlebenchmark GIT_REPOSITORY
                                                                    https : // github.com/google/benchmark.git
                                                                GIT_TAG
                                                                    v1.7.1) FetchContent_MakeAvailable(googlebenchmark) endif()

                                                                if (T81LIB_BUILD_TESTS) enable_testing() add_subdirectory(
                                                                    tests) endif()

                                                                    if (T81LIB_BUILD_BENCH) add_subdirectory(
                                                                        bench) endif()

                                                                        if (T81LIB_BUILD_PYTHON_BINDINGS) add_subdirectory(
                                                                            python) endif()

                                                                            add_subdirectory(
                                                                                examples)

                                                                                include(
                                                                                    GNUInstallDirs)

                                                                                    install(
                                                                                        TARGETS t81lib
                                                                                            EXPORT
                                                                                                t81libTargets)

                                                                                        install(
                                                                                            DIRECTORY
                                                                                                include /
                                                                                            t81 DESTINATION
                                                                                                ${CMAKE_INSTALL_INCLUDEDIR})

                                                                                            install(
                                                                                                EXPORT t81libTargets NAMESPACE
                                                                                                    t81::DESTINATION
                                                                                                        ${CMAKE_INSTALL_LIBDIR} /
                                                                                                cmake /
                                                                                                t81lib)

                                                                                                include(
                                                                                                    CMakePackageConfigHelpers)

                                                                                                    configure_package_config_file(
                                                                                                        cmake /
                                                                                                        t81libConfig
                                                                                                            .cmake
                                                                                                            .in ${
                                                                                                                CMAKE_CURRENT_BINARY_DIR} /
                                                                                                        t81libConfig
                                                                                                            .cmake INSTALL_DESTINATION
                                                                                                                ${CMAKE_INSTALL_LIBDIR} /
                                                                                                        cmake /
                                                                                                        t81lib)

                                                                                                        write_basic_package_version_file(
                                                                                                            ${CMAKE_CURRENT_BINARY_DIR} /
                                                                                                            t81libConfigVersion
                                                                                                                .cmake VERSION
                                                                                                                    ${PROJECT_VERSION} COMPATIBILITY
                                                                                                                        SameMajorVersion)

                                                                                                            install(
                                                                                                                FILES ${
                                                                                                                    CMAKE_CURRENT_BINARY_DIR} /
                                                                                                                t81libConfig
                                                                                                                    .cmake ${
                                                                                                                        CMAKE_CURRENT_BINARY_DIR} /
                                                                                                                t81libConfigVersion
                                                                                                                    .cmake DESTINATION
                                                                                                                        ${CMAKE_INSTALL_LIBDIR} /
                                                                                                                cmake /
                                                                                                                t81lib)
