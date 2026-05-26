cmake_minimum_required(VERSION 3.10)

# This lesson shows how to read and display an image with OpenCV.
project(opencv_read_show_image LANGUAGES CXX)

# Find the OpenCV package installed on the system.
find_package(OpenCV REQUIRED)

# Build the lesson executable from main.cpp.
add_executable(opencv_read_show_image main.cpp)

# Ask CMake to use the C++17 standard for this target.
target_compile_features(opencv_read_show_image PRIVATE cxx_std_17)

# Disable compiler-specific language extensions, so the code stays closer to standard C++17.
set_target_properties(opencv_read_show_image PROPERTIES
    CXX_EXTENSIONS OFF
)

# Add OpenCV header paths and libraries to this target.
target_include_directories(opencv_read_show_image PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_read_show_image PRIVATE ${OpenCV_LIBS})
