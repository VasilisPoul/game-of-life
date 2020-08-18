#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

#define TABLE_N 20
#define TABLE_M 20

bool **allocate2DArray(int rows, int columns);

void Free2DArray(bool **array, int rows);

int **create(int n, int m);

void initialize_array(bool **array, int n, int m);

void print_array(bool **array, int rowDim, int colDim, int localRowDim, int localColDim);

int operate(bool **array, int n, int m);

#endif
