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

add_library(lesson31_kernel_variants src/kernel_variants.cu)
target_compile_features(lesson31_kernel_variants PUBLIC cxx_std_17)
target_include_directories(lesson31_kernel_variants PUBLIC include)
target_link_libraries(lesson31_kernel_variants PUBLIC CUDA::cudart CUDA::nvtx3)
set_target_properties(lesson31_kernel_variants PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)

add_executable(lesson31_kernel_benchmark src/main.cpp)
target_compile_features(lesson31_kernel_benchmark PRIVATE cxx_std_17)
target_link_libraries(lesson31_kernel_benchmark PRIVATE lesson31_kernel_variants)

if(BUILD_TESTING)
    add_executable(lesson31_kernel_tests tests/test_kernel_variants.cpp)
    target_compile_features(lesson31_kernel_tests PRIVATE cxx_std_17)
    target_link_libraries(lesson31_kernel_tests PRIVATE lesson31_kernel_variants)
    add_test(NAME lesson31_kernel_variants COMMAND lesson31_kernel_tests)
    set_tests_properties(lesson31_kernel_variants PROPERTIES SKIP_RETURN_CODE 77)

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME lesson31_profile_tools
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p test_*.py -v
    )
endif()
