cmake_minimum_required(VERSION 3.18)
project(cpp_shared_library_python_binding LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInferRuntime.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu REQUIRED)

add_library(trt_inference SHARED
    src/trt_c_api.cpp
    ../17_dynamic_batching/src/batch_layout.cpp
    ../17_dynamic_batching/src/dynamic_batch_runner.cpp)
target_compile_features(trt_inference PRIVATE cxx_std_17)
target_include_directories(trt_inference PUBLIC include
    PRIVATE ../17_dynamic_batching/include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_inference PRIVATE CUDA::cudart ${TENSORRT_LIBRARY})
set_target_properties(trt_inference PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
