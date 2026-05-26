cmake_minimum_required(VERSION 3.10)

# Generate compile_commands.json for IDEs such as VSCode.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# This project is a very small C++ program for the first lesson.
project(hello_world LANGUAGES CXX)

# Build the program from main.cpp.
add_executable(hello_world main.cpp)

# Ask CMake to use the C++17 standard for this target.
target_compile_features(hello_world PRIVATE cxx_std_17)

# Disable compiler-specific language extensions, so the code stays closer to standard C++17.
set_target_properties(hello_world PROPERTIES
    CXX_EXTENSIONS OFF
)
