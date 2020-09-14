#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

#include <stdbool.h>

char **allocate2DArray(int rows, int columns);

void free2DArray(char **block, int rows);

void initialize_block(char **block, bool zeroFill, int n, int m);

void print_array(char **array, bool split, bool internals, int rowDim, int colDim, int localRowDim, int localColDim);

void calculate(char **old, char **current, int i, int j, int *changes);

#endif
