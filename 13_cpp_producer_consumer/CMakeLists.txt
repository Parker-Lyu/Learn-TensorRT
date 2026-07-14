cmake_minimum_required(VERSION 3.18)

project(cpp_producer_consumer LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
find_package(Threads REQUIRED)

option(ENABLE_TSAN "Build CPU-only targets with ThreadSanitizer" OFF)

add_library(producer_consumer_pipeline
    src/image_pipeline.cpp
)
target_compile_features(producer_consumer_pipeline PUBLIC cxx_std_17)
target_include_directories(producer_consumer_pipeline PUBLIC include)
target_link_libraries(producer_consumer_pipeline PUBLIC Threads::Threads)
set_target_properties(producer_consumer_pipeline PROPERTIES CXX_EXTENSIONS OFF)

add_executable(cpp_producer_consumer src/main.cpp)
target_compile_features(cpp_producer_consumer PRIVATE cxx_std_17)
target_link_libraries(cpp_producer_consumer PRIVATE producer_consumer_pipeline)
set_target_properties(cpp_producer_consumer PROPERTIES CXX_EXTENSIONS OFF)

if(BUILD_TESTING)
    add_executable(producer_consumer_tests tests/test_producer_consumer.cpp)
    target_compile_features(producer_consumer_tests PRIVATE cxx_std_17)
    target_link_libraries(producer_consumer_tests PRIVATE producer_consumer_pipeline)
    set_target_properties(producer_consumer_tests PROPERTIES CXX_EXTENSIONS OFF)
    add_test(NAME producer_consumer_tests COMMAND producer_consumer_tests)
endif()

if(ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        foreach(target producer_consumer_pipeline cpp_producer_consumer producer_consumer_tests)
            if(TARGET ${target})
                target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer -g)
                target_link_options(${target} PRIVATE -fsanitize=thread)
            endif()
        endforeach()
    else()
        message(FATAL_ERROR "ENABLE_TSAN requires GCC or Clang")
    endif()
endif()
