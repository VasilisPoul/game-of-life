#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

#define TABLE_N 12
#define TABLE_M 12

bool **allocate2DArray(int rows, int columns);

void free2DArray(bool **array, int rows);

int **create(int n, int m);

void initialize_array(bool **array, int n, int m);

void print_array(bool **array, bool split, bool internals, int rowDim, int colDim, int localRowDim, int localColDim);

void calculate(bool **a, bool **b, int i, int j, int *changes);

int operate(bool **array, int n, int m);

#endif
