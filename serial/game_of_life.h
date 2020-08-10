#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

#define N 10
#define M 10

int **create(int n, int m);

void initialize(int array[N][M], int n, int m);

void print_array(int array[N][M], int n, int m);

int operate(int array[N][M], int n, int m);

#endif
