set(OPT_FLAGS, "-Ofast")

set(
    SRC_CPU
    src/image/image.cu
    src/block/blocks.cu
    src/block/block.cu
)

include_directories(
    src
    src/image
    src/block
)

cmake_minimum_required(VERSION 3.12)
project(LBP LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

add_executable(lbp)
target_sources(lbp PRIVATE "src/main.cu" ${SRC_CPU})

target_compile_options(lbp PRIVATE -Wall -Wextra)

set_target_properties(lbp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(lbp PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
