cmake_minimum_required(VERSION 2.6)
project(indexdesc_benchmark)

set(CMAKE_CXX_STANDARD 98)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Za /D_CRT_SECURE_NO_WARNINGS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra -Werror")
endif()


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Flann REQUIRED)

include_directories(SYSTEM ${FLANN_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(indexdesc_benchmark ${SOURCE_FILES})

target_link_libraries(indexdesc_benchmark ${FLANN_LIBRARIES})