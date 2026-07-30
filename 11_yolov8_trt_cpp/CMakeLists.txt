cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(yolov8_trt_cpp LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

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

add_library(yolov8_preprocess
    src/preprocess.cpp
)
target_compile_features(yolov8_preprocess PUBLIC cxx_std_17)
set_target_properties(yolov8_preprocess PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(yolov8_preprocess PUBLIC include)
target_link_libraries(yolov8_preprocess PUBLIC opencv_core opencv_imgproc)

add_library(yolov8_postprocess
    src/postprocess.cpp
)
target_compile_features(yolov8_postprocess PUBLIC cxx_std_17)
set_target_properties(yolov8_postprocess PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(yolov8_postprocess PUBLIC include)
target_link_libraries(yolov8_postprocess PUBLIC opencv_core)

add_library(yolov8_trt_runtime
    src/tensorrt_runner.cpp
)
target_compile_features(yolov8_trt_runtime PUBLIC cxx_std_17)
set_target_properties(yolov8_trt_runtime PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(yolov8_trt_runtime
    PUBLIC include
    PRIVATE ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(yolov8_trt_runtime
    PUBLIC CUDA::cudart CUDA::nvtx3
    PRIVATE ${TENSORRT_LIBRARY}
)

add_library(yolov8_visualize
    src/visualize.cpp
)
target_compile_features(yolov8_visualize PUBLIC cxx_std_17)
set_target_properties(yolov8_visualize PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(yolov8_visualize PUBLIC include)
target_link_libraries(yolov8_visualize PUBLIC opencv_core opencv_imgproc)

add_executable(yolov8_trt_cpp
    src/main.cpp
)
target_compile_features(yolov8_trt_cpp PRIVATE cxx_std_17)
set_target_properties(yolov8_trt_cpp PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(yolov8_trt_cpp
    PRIVATE
        yolov8_preprocess
        yolov8_postprocess
        yolov8_trt_runtime
        yolov8_visualize
        opencv_imgcodecs
)

add_executable(yolov8_cpp_tests
    tests/test_preprocess_postprocess.cpp
)
target_compile_features(yolov8_cpp_tests PRIVATE cxx_std_17)
set_target_properties(yolov8_cpp_tests PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(yolov8_cpp_tests
    PRIVATE
        yolov8_preprocess
        yolov8_postprocess
        opencv_core
        opencv_imgproc
)
