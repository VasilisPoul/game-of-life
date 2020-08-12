cmake_minimum_required(VERSION 3.16)
project(game_of_life C)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(SOURCE_FILES mpi/main.c)

add_executable(game_of_life ${SOURCE_FILES})

target_link_libraries(game_of_life m)
