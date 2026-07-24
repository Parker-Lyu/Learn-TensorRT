cmake_minimum_required(VERSION 3.18)

# This project is a very small C++ program for the first lesson.
project(hello_world LANGUAGES CXX)

# Generate compile_commands.json for editors and static-analysis tools.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build the program from main.cpp.
add_executable(hello_world main.cpp)

# Require C++17 for this target rather than allowing CMake to fall back to an older standard.
target_compile_features(hello_world PRIVATE cxx_std_17)

# Disable compiler-specific language extensions, so the code stays closer to standard C++17.
set_target_properties(hello_world PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
