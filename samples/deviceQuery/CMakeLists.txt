cmake_minimum_required(VERSION 3.18) # CUDA 架构支持需要较新的 CMake 版本
project(DeviceQuery LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# native 选择本机 GPU 架构，此特殊值要求 CMake 3.24 或更高版本。
# all 从 3.23 起可用；显式架构列表采用 80;86;89 这类整数编号，而非带小数点的写法。
# 文件声明的最低版本仍为 3.18，本次只说明兼容条件，不改变构建配置。
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(deviceQuery deviceQuery.cpp)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(deviceQuery PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()


target_link_libraries(deviceQuery PRIVATE CUDA::cudart)

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
