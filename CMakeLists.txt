cmake_minimum_required(VERSION 3.16)
project(game_of_life C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(game_of_life game_of_life.c main.c)
