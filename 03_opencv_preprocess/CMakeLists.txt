cmake_minimum_required(VERSION 3.10)

# Generate compile_commands.json for IDEs such as VSCode.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(opencv_preprocess LANGUAGES CXX)

find_package(OpenCV REQUIRED)

add_executable(opencv_preprocess
    main.cpp
    preprocess.cpp
)

target_compile_features(opencv_preprocess PRIVATE cxx_std_17)

set_target_properties(opencv_preprocess PROPERTIES
    CXX_EXTENSIONS OFF
)

target_include_directories(opencv_preprocess PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_preprocess PRIVATE ${OpenCV_LIBS})
