#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

#include <stdbool.h>

#define N 10
#define M 10

char **cudaAllocate2DArray(int rows, int columns);

char **allocate2DArray(int rows, int columns);

void Free2DArray(char **array, int rows);

void initialize(char **array, int n, int m);

void print_array(char **array, int n, int m);

int operate(char **array, int n, int m);

#endif
