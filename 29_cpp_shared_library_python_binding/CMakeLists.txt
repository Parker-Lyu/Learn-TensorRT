cmake_minimum_required(VERSION 3.18)
project(cpp_shared_library_python_binding LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInferRuntime.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu REQUIRED)

add_library(trt_core STATIC
    ../17_dynamic_batching/src/batch_layout.cpp
    ../17_dynamic_batching/src/dynamic_batch_runner.cpp)
target_compile_features(trt_core PRIVATE cxx_std_17)
target_include_directories(trt_core PUBLIC ../17_dynamic_batching/include
    PRIVATE ../17_dynamic_batching/include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_core PUBLIC CUDA::cudart ${TENSORRT_LIBRARY})
set_target_properties(trt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(trt_inference SHARED src/trt_c_api.cpp)
target_compile_features(trt_inference PRIVATE cxx_std_17)
target_include_directories(trt_inference PUBLIC include)
target_link_libraries(trt_inference PRIVATE trt_core)
set_target_properties(trt_inference PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(trt_inference_py MODULE src/trt_pybind.cpp)
target_compile_features(trt_inference_py PRIVATE cxx_std_17)
target_link_libraries(trt_inference_py PRIVATE trt_core)
