cmake_minimum_required(VERSION 3.18)
project(cpp_interview_katas LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(cv_katas src/katas.cpp)
target_compile_features(cv_katas PUBLIC cxx_std_17)
target_include_directories(cv_katas PUBLIC include)
target_link_libraries(cv_katas PUBLIC Threads::Threads)

add_library(cuda_buffer INTERFACE)
target_compile_features(cuda_buffer INTERFACE cxx_std_17)
target_include_directories(cuda_buffer INTERFACE include)
target_link_libraries(cuda_buffer INTERFACE CUDA::cudart)

add_executable(katas_demo src/main.cpp)
target_compile_features(katas_demo PRIVATE cxx_std_17)
target_link_libraries(katas_demo PRIVATE cv_katas)

if(BUILD_TESTING)
    add_executable(katas_tests tests/test_katas.cpp)
    target_compile_features(katas_tests PRIVATE cxx_std_17)
    target_link_libraries(katas_tests PRIVATE cv_katas)
    add_test(NAME katas_tests COMMAND katas_tests)

    add_executable(cuda_buffer_tests tests/test_cuda_buffer.cpp)
    target_compile_features(cuda_buffer_tests PRIVATE cxx_std_17)
    target_link_libraries(cuda_buffer_tests PRIVATE cuda_buffer)
    add_test(NAME cuda_buffer_tests COMMAND cuda_buffer_tests)
    set_tests_properties(cuda_buffer_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
