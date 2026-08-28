cmake_minimum_required(VERSION 3.18)
project(nsight_compute_kernel_analysis LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

include(CTest)
find_package(CUDAToolkit REQUIRED)

add_library(lesson31_mlp src/mlp_layernorm.cu)
target_compile_features(lesson31_mlp PUBLIC cxx_std_17)
target_include_directories(lesson31_mlp PUBLIC include)
target_link_libraries(lesson31_mlp PUBLIC CUDA::cublas CUDA::cudart CUDA::nvtx3)
set_target_properties(lesson31_mlp PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)

add_executable(lesson31_mlp_benchmark src/benchmark_main.cpp)
target_compile_features(lesson31_mlp_benchmark PRIVATE cxx_std_17)
target_link_libraries(lesson31_mlp_benchmark PRIVATE lesson31_mlp)

if(BUILD_TESTING)
    add_executable(lesson31_mlp_tests tests/test_mlp_layernorm.cpp)
    target_compile_features(lesson31_mlp_tests PRIVATE cxx_std_17)
    target_link_libraries(lesson31_mlp_tests PRIVATE lesson31_mlp)
    add_test(NAME lesson31_mlp_layernorm COMMAND lesson31_mlp_tests)
    set_tests_properties(lesson31_mlp_layernorm PROPERTIES SKIP_RETURN_CODE 77)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME lesson31_profile_tools
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_*.py -v
    )
endif()
