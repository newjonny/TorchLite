cmake_minimum_required(VERSION 3.14)

project(TorchLite VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mlp_network
    src/main.cpp
    src/layer.cpp
    src/value.cpp
)

# where to find the header files
target_include_directories(mlp_network PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# strict compiler warnings
target_compile_options(mlp_network PRIVATE -Wall -Wextra -Wpedantic)