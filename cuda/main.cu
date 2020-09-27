#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <cuda.h>
#include "game_of_life.h"

void cudaAllocate2DArray(char **block, int rows, int columns) {
    int i;
    cudaMalloc((void ***) &block, rows * sizeof(char *));
//    block = (char **) malloc(rows * sizeof(char *));
    cudaMalloc((void **) &block[0], rows * columns * sizeof(char));
    //block[0] = (char *) malloc(rows * columns * sizeof(char));
    for (i = 1; i < rows; i++) {
        block[i] = &(block[0][i * rows]);
    }
    memset(block[0], (int) '0', rows * columns * sizeof(char));
    return;
}

char **allocate2DArray(int rows, int columns) {
    char **block;
    int i;
    block = (char **) malloc(rows * sizeof(char *));
    block[0] = (char *) malloc(rows * columns * sizeof(char));
    for (i = 1; i < rows; i++) {
        block[i] = &(block[0][i * rows]);
    }
    memset(block[0], (int) '0', rows * columns * sizeof(char));
    return block;
}

int main(int argc, char **argv) {
    char **block;
    int rows = 0,
    int columns = 0;
//    block = allocate2DArray(N, M);
//    print_array(block, N, M);
//    initialize(block, N, M);





    cudaAllocate2DArray<<<1, 1>>>(block, rows, columns);


//    while (operate(array, N, M));
//
//    Free2DArray(array, N);
}
