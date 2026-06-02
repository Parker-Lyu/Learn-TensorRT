cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(opencv_preprocess_practice LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_executable(opencv_preprocess_practice
    src/main.cpp
    src/preprocess.cpp
)

target_compile_features(opencv_preprocess_practice PRIVATE cxx_std_17)

set_target_properties(opencv_preprocess_practice PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(opencv_preprocess_practice
    PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(opencv_preprocess_practice PRIVATE ${OpenCV_LIBS})
