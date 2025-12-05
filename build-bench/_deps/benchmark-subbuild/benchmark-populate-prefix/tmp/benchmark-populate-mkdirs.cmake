# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/t81dev/t81lib/build-bench/_deps/benchmark-src")
  file(MAKE_DIRECTORY "/Users/t81dev/t81lib/build-bench/_deps/benchmark-src")
endif()
file(MAKE_DIRECTORY
  "/Users/t81dev/t81lib/build-bench/_deps/benchmark-build"
  "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix"
  "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix/tmp"
  "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp"
  "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix/src"
  "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/t81dev/t81lib/build-bench/_deps/benchmark-subbuild/benchmark-populate-prefix/src/benchmark-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
