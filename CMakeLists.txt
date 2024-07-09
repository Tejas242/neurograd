cmake_minimum_required(VERSION 3.12.0)
project(neurograd VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create the library
add_library(neurograd
    neurograd.cpp
    nn.h
)

# Set include directories for the library
target_include_directories(neurograd PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Include the engine directory for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Testing
include(CTest)
enable_testing()

add_subdirectory(tests)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)