project(npuemul CXX)
cmake_minimum_required(VERSION 3.0)

enable_testing()
include(ExternalProject)
include(GoogleTest)

if(UNIX)
    set(OS_SPEC_LIBS "pthread")
    set(SRC_OS_SPEC_FOLDER unix)
elseif(WIN32)
    set(SRC_OS_SPEC_FOLDER win)
    set(OS_SPEC_LIBS shlwapi)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /std:c++latest /Zi)
else()
    add_compile_options(-O3 -g -march=native -ftree-vectorize)
endif()

set(ASM_OBJ src/${SRC_OS_SPEC_FOLDER}/MatmulMicrokernel.obj)

include_directories(include)
file(GLOB NPUEMUL_SRC "src/*.cpp")
add_library(npuemul ${ASM_OBJ} ${NPUEMUL_SRC})

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG main
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_GMOCK=OFF -Dgtest_force_shared_crt=ON -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

file(GLOB TESTS_SRC "tests/*.cpp")
add_executable(tests ${TESTS_SRC})
add_dependencies(tests googletest)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_libraries(tests npuemul gtest_main gtest ${OS_SPEC_LIBS})
else()
    target_link_libraries(tests npuemul gtest_maind gtestd  ${OS_SPEC_LIBS})
endif()

file(GLOB BENCHMARKS_SRC "benchmarks/*.cpp")
add_executable(benchmarks ${BENCHMARKS_SRC})
add_dependencies(benchmarks googlebenchmark)
target_link_libraries(benchmarks npuemul benchmark_main benchmark ${OS_SPEC_LIBS})
gtest_discover_tests(tests)
