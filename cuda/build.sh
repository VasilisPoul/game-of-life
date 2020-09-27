#!/bin/bash

nvcc -o game_of_life main.cu game_of_life.cu

./game_of_life
