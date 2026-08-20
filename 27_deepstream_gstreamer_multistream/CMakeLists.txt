cmake_minimum_required(VERSION 3.18)
project(deepstream_yolov8_parser LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(DEEPSTREAM_INCLUDE_DIR nvdsinfer_custom_impl.h
    PATHS /opt/nvidia/deepstream/deepstream/sources/includes)
if(NOT DEEPSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR "DeepStream headers not found; build inside the documented DeepStream container")
endif()

# DeepStream's C++ headers include CUDA runtime types, but the host compiler does not
# search the CUDA toolkit include directory unless the project adds it explicitly.
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS
        "${CUDAToolkit_ROOT}"
        "$ENV{CUDA_HOME}"
        /usr/local/cuda
        /usr/local/cuda-12.8
    PATH_SUFFIXES include targets/x86_64-linux/include)
if(NOT CUDA_INCLUDE_DIR)
    message(FATAL_ERROR
        "CUDA headers not found; build inside the documented DeepStream container "
        "or set CUDAToolkit_ROOT/CUDA_HOME")
endif()

add_library(nvdsinfer_custom_yolov8 SHARED src/yolov8_parser.cpp)
target_compile_features(nvdsinfer_custom_yolov8 PRIVATE cxx_std_17)
target_include_directories(nvdsinfer_custom_yolov8 PRIVATE
    ${DEEPSTREAM_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR})
set_target_properties(nvdsinfer_custom_yolov8 PROPERTIES CXX_EXTENSIONS OFF)
