cmake_minimum_required(VERSION 3.10)

# Generate compile_commands.json for IDEs such as VSCode.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(opencv_preprocess LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_library(preprocess STATIC
    src/preprocess.cpp
)

target_compile_features(preprocess PUBLIC cxx_std_17)

set_target_properties(preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(preprocess
    PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(preprocess PUBLIC ${OpenCV_LIBS})

add_executable(opencv_preprocess src/main.cpp)
target_compile_features(opencv_preprocess PRIVATE cxx_std_17)
set_target_properties(opencv_preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_link_libraries(opencv_preprocess PRIVATE preprocess)

include(CTest)
if(BUILD_TESTING)
    add_executable(preprocess_tests tests/preprocess_tests.cpp)
    target_compile_features(preprocess_tests PRIVATE cxx_std_17)
    set_target_properties(preprocess_tests PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    target_link_libraries(preprocess_tests PRIVATE preprocess)
    add_test(NAME preprocess_tests COMMAND preprocess_tests)
endif()
