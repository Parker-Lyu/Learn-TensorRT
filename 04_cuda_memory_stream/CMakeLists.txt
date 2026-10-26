cmake_minimum_required(VERSION 3.18)

# Generate compile_commands.json for IDEs such as VSCode.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cuda_memory_stream LANGUAGES CXX)

include(CTest)
option(CPU_ONLY_TESTS "Build transform host checks without CUDA" OFF)
if(BUILD_TESTING)
    add_executable(transform_checks_tests tests/test_transform_checks.cpp)
    target_compile_features(transform_checks_tests PRIVATE cxx_std_17)
    set_target_properties(transform_checks_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(transform_checks_tests PRIVATE include)
    add_test(NAME transform_checks_tests COMMAND transform_checks_tests)
endif()
if(CPU_ONLY_TESTS)
    return()
endif()
enable_language(CUDA)

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
