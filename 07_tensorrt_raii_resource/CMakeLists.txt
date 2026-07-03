cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tensorrt_raii_resource LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInferRuntime.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
    REQUIRED
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/lib
    REQUIRED
)

add_library(tensorrt_raii_lib
    src/tensorrt_raii.cpp
)

target_compile_features(tensorrt_raii_lib PUBLIC cxx_std_17)

set_target_properties(tensorrt_raii_lib PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(tensorrt_raii_lib
    PUBLIC
        include
    PRIVATE
        ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(tensorrt_raii_lib
    PUBLIC
        CUDA::cudart
    PRIVATE
        ${TENSORRT_LIBRARY}
)

add_executable(tensorrt_raii_resource
    src/main.cpp
)

target_compile_features(tensorrt_raii_resource PRIVATE cxx_std_17)

set_target_properties(tensorrt_raii_resource PROPERTIES
    CXX_EXTENSIONS OFF
)

target_link_libraries(tensorrt_raii_resource PRIVATE tensorrt_raii_lib)
