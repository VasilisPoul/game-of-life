#ifndef __GAME_OF_LIFE_H__
#define __GAME_OF_LIFE_H__

int **create(int n, int m);

void initialize(int **array, int n, int m);

void print_array(int **array, int n, int m);

int operate(int **array, int n, int m);

#endif