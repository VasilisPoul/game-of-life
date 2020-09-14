cmake_minimum_required(VERSION 3.16)
project(game_of_life C)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

#set(SOURCE_FILES mpi/main.c mpi/mpi.c mpi/game_of_life.c)
set(SOURCE_FILES mpi+openmp/main.c mpi+openmp/mpi.c mpi+openmp/game_of_life.c)
#set(SOURCE_FILES serial/main.c serial/game_of_life.c)

#add_compile_options(-g -c -Wall -O3)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O3")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(game_of_life ${SOURCE_FILES})

target_link_libraries(game_of_life m)
