cmake_minimum_required(VERSION 3.16)
project(alphago_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags for release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Use pybind11's FindPython
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Build the Python module
pybind11_add_module(alphago_core
    src/board.cpp
    src/mcts.cpp
    src/features.cpp
    src/bindings.cpp
)

target_include_directories(alphago_core PRIVATE src)

# Install to project root so Python can import it
install(TARGETS alphago_core DESTINATION ${CMAKE_SOURCE_DIR})
