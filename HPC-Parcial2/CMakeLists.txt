cmake_minimum_required(VERSION 3.0)

project(MultMatrix)
if(NOT CMAKE_BUILD_TYPE)
     set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-Wno-deprecated-gpu-targets")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(Mult_Cuda Mult_Cuda.cu)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
add_executable(Main Main.cpp)
target_link_libraries(Main ${MPI_CXX_LIBRARIES} Mult_Cuda)