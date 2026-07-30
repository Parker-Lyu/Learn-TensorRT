cmake_minimum_required(VERSION 3.18)
project(deepstream_yolov8_parser LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(DEEPSTREAM_INCLUDE_DIR nvdsinfer_custom_impl.h
    PATHS /opt/nvidia/deepstream/deepstream/sources/includes)
if(NOT DEEPSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR "DeepStream headers not found; build inside the documented DeepStream container")
endif()

add_library(nvdsinfer_custom_yolov8 SHARED src/yolov8_parser.cpp)
target_compile_features(nvdsinfer_custom_yolov8 PRIVATE cxx_std_17)
target_include_directories(nvdsinfer_custom_yolov8 PRIVATE ${DEEPSTREAM_INCLUDE_DIR})
set_target_properties(nvdsinfer_custom_yolov8 PROPERTIES CXX_EXTENSIONS OFF)
