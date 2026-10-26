cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tensorrt_cpp_basic LANGUAGES CXX)

include(CTest)
option(CPU_ONLY_TESTS "Build tensor storage tests without CUDA or TensorRT" OFF)
if(BUILD_TESTING)
    add_executable(tensor_size_tests tests/test_tensor_size.cpp)
    target_compile_features(tensor_size_tests PRIVATE cxx_std_17)
    set_target_properties(tensor_size_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(tensor_size_tests PRIVATE include)
    add_test(NAME tensor_size_tests COMMAND tensor_size_tests)
endif()
if(CPU_ONLY_TESTS)
    return()
endif()

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h NvOnnxParser.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
    REQUIRED
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

find_library(TENSORRT_ONNX_PARSER_LIBRARY
    NAMES nvonnxparser
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

add_library(tensorrt_cpp_basic_lib
    src/tensorrt_basic.cpp
)

target_compile_features(tensorrt_cpp_basic_lib PUBLIC cxx_std_17)

set_target_properties(tensorrt_cpp_basic_lib PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(tensorrt_cpp_basic_lib
    PUBLIC
        include
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(tensorrt_cpp_basic_lib
    PUBLIC
        CUDA::cudart
    PRIVATE
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
)

add_executable(tensorrt_cpp_basic
    src/main.cpp
)

target_compile_features(tensorrt_cpp_basic PRIVATE cxx_std_17)

set_target_properties(tensorrt_cpp_basic PROPERTIES
    CXX_EXTENSIONS OFF
)

target_link_libraries(tensorrt_cpp_basic PRIVATE tensorrt_cpp_basic_lib)
