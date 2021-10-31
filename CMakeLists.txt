project(npuemul C CXX)
cmake_minimum_required(VERSION 3.21)

include(ExternalProject)

set(ASM_SRC src/MatmulMicrokernel.obj)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-O3 -g -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/arch:AVX2)
endif()

include_directories(include)
add_library(npuemul ${ASM_SRC} src/Matmul.cpp src/Conv2D.cpp src/ReLu.cpp src/MaxPool2D.cpp src/Dense.cpp
    src/Threads.cpp src/Types.cpp src/Memory.cpp)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

ExternalProject_Add(googletest
    GIT_REPOSITORY https://github.com/google/googletest
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_GMOCK=OFF -Dgtest_force_shared_crt=ON -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

add_executable(tests tests/TestsMain.cpp tests/MatmulTests.cpp tests/ReLuTests.cpp tests/DenseTests.cpp tests/Conv2DTests.cpp)
add_dependencies(tests googletest)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_libraries(tests npuemul gtest_main gtest)
else()
    target_link_libraries(tests npuemul gtest_maind gtestd)
endif()

add_executable(benchmarks benchmarks/MatmulBenchmarks.cpp benchmarks/Conv2DBenchmarks.cpp
    benchmarks/ReLuBenchmarks.cpp benchmarks/MaxPool2DBenchmarks.cpp benchmarks/DenseBenchmarks.cpp
    benchmarks/VGG16Benchmarks.cpp)
add_dependencies(benchmarks googlebenchmark)
target_link_libraries(benchmarks npuemul benchmark_main benchmark shlwapi)
