cmake_minimum_required(VERSION 3.18)
project(custom_tensorrt_plugin LANGUAGES CXX CUDA)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInferRuntime.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
find_library(TENSORRT_LIBRARY NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu REQUIRED)

add_library(acme_swish_plugin SHARED src/acme_swish_plugin.cu)
target_compile_features(acme_swish_plugin PUBLIC cxx_std_17)
target_include_directories(acme_swish_plugin PUBLIC include PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(acme_swish_plugin PRIVATE CUDA::cudart ${TENSORRT_LIBRARY})
set_target_properties(acme_swish_plugin PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)

add_executable(validate_plugin src/validate_plugin.cpp)
target_compile_features(validate_plugin PRIVATE cxx_std_17)
target_include_directories(validate_plugin PRIVATE include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(validate_plugin PRIVATE acme_swish_plugin CUDA::cudart ${TENSORRT_LIBRARY})
