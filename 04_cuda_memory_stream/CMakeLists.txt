cmake_minimum_required(VERSION 3.18)

# Generate compile_commands.json for IDEs such as VSCode.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cuda_memory_stream LANGUAGES CXX CUDA)

add_executable(cuda_memory_stream
    src/main.cpp
    src/cuda_memory_demo.cu
)

target_compile_features(cuda_memory_stream PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(cuda_memory_stream PROPERTIES
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(cuda_memory_stream PRIVATE include)
