#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

#include <stdbool.h>

#define N 10
#define M 10

bool **allocate2DArray(int rows, int columns);

void Free2DArray(bool **array, int rows);

void initialize(bool **array, int n, int m);

void print_array(bool **array, int n, int m);

int operate(bool **array, int n, int m);

#endif
