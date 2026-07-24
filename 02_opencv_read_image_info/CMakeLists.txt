cmake_minimum_required(VERSION 3.10)

# Generate compile_commands.json for IDEs such as VSCode.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This lesson shows how to read an image and inspect its metadata with OpenCV.
project(opencv_read_image_info LANGUAGES CXX)

# This lesson only needs OpenCV's matrix and image-codec modules. Listing the
# components keeps the dependency boundary visible to learners.
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

# Build the lesson executable from main.cpp.
add_executable(opencv_read_image_info main.cpp)

# Ask CMake to use the C++17 standard for this target. The explicit standard
# properties ensure GCC uses -std=c++17 rather than its GNU-extension default.
target_compile_features(opencv_read_image_info PRIVATE cxx_std_17)

# Disable compiler-specific language extensions, so the code stays closer to standard C++17.
set_target_properties(opencv_read_image_info PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add OpenCV header paths and libraries to this target.
target_include_directories(opencv_read_image_info PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_read_image_info PRIVATE ${OpenCV_LIBS})
