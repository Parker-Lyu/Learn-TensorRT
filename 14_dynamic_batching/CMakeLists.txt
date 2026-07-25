cmake_minimum_required(VERSION 3.18)
project(dynamic_batching LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CTest)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NAMES NvInferRuntime.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib REQUIRED)

add_library(batch_layout src/batch_layout.cpp)
target_compile_features(batch_layout PUBLIC cxx_std_17)
target_include_directories(batch_layout PUBLIC include)

add_library(dynamic_batch_runner src/dynamic_batch_runner.cpp)
target_compile_features(dynamic_batch_runner PUBLIC cxx_std_17)
target_include_directories(dynamic_batch_runner PUBLIC include PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(dynamic_batch_runner PUBLIC batch_layout CUDA::cudart PRIVATE ${TENSORRT_LIBRARY})

add_executable(dynamic_batching src/main.cpp)
target_compile_features(dynamic_batching PRIVATE cxx_std_17)
target_link_libraries(dynamic_batching PRIVATE dynamic_batch_runner)

if(BUILD_TESTING)
    add_executable(batch_layout_tests tests/test_batch_layout.cpp)
    target_compile_features(batch_layout_tests PRIVATE cxx_std_17)
    target_link_libraries(batch_layout_tests PRIVATE batch_layout)
    add_test(NAME batch_layout_tests COMMAND batch_layout_tests)
endif()
