cmake_minimum_required(VERSION 3.16)
project(raytrace LANGUAGES CXX)

add_executable(raytrace src/main.cpp)
target_compile_features(raytrace PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(raytrace PRIVATE -Wall -Wextra -Wpedantic)
endif()
