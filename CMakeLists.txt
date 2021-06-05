set(OPT_FLAGS, "-Ofast")

set(
    SRC_CPU
)

include_directories(
    src
)

cmake_minimum_required(VERSION 3.12)
project(LBP)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_executable(lbp)
target_sources(lbp PRIVATE "src/main.cc" ${SRC_CPU})

target_compile_options(lbp PRIVATE -Wall -Wextra)
