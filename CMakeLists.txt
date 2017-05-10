cmake_minimum_required(VERSION 2.6)
project(indexdesc_benchmark)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Flann REQUIRED)

include_directories(${FLANN_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(indexdesc_benchmark ${SOURCE_FILES})

target_link_libraries(indexdesc_benchmark ${FLANN_LIBRARIES})