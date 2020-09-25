#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "game_of_life.h"

int mod(int x, int m) {
    return (x % m + m) % m;
}

__global__ char **cudaAllocate2DArray(int rows, int columns) {
    char **block;
    int i;
    cudaMalloc((void ***) &block , rows * sizeof(char *));
//    block = (char **) malloc(rows * sizeof(char *));
    cudaMalloc((void **) &block[0], rows * columns * sizeof(char));
    //block[0] = (char *) malloc(rows * columns * sizeof(char));
    for (i = 1; i < rows; i++) {
        block[i] = &(block[0][i * rows]);
    }
    memset(block[0], (int) '0', rows * columns * sizeof(char));
    return block;
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

void Free2DArray(char **block, int rows) {
    free(block[0]);
    free(block);
}

void print_array(char **array, int n, int m) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%s ", array[i][j] == true ? "\u2B1B" : "\u2B1C");
        }
        printf("\n");
    }
}

void initialize(char **array, int n, int m) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array[i][j] = rand() % 2;
        }
    }
    print_array(array, n, m);
}

void copy(char **target, char **source, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            target[i][j] = source[i][j];
        }
    }
}

int operate(char **array, int n, int m) {
    char **old_array = allocate2DArray(N, M);
    int left = 0, right = 0;
    int up = 0, down = 0;
    int up_left = 0, up_right = 0;
    int down_left = 0, down_right = 0;
    int changes = 0;
    copy(old_array, array, n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            left = old_array[i][mod(j - 1, m)];
            right = old_array[i][mod(j + 1, m)];

            up = old_array[mod(i - 1, n)][j];
            down = old_array[mod(i + 1, n)][j];

            up_left = old_array[mod(i - 1, n)][mod(j - 1, m)];
            up_right = old_array[mod(i - 1, n)][mod(j + 1, m)];

            down_left = old_array[mod(i + 1, n)][mod(j - 1, m)];
            down_right = old_array[mod(i + 1, n)][mod(j + 1, m)];

            int sum = left + right + up_left + up_right + down_left + down_right + up + down;

            if (old_array[i][j]) {
                if (sum <= 1 || sum >= 4) {
                    array[i][j] = 0;
                    changes++;
                }
            } else if (sum == 3) {
                array[i][j] = 1;
                changes++;
            }
        }
    }
    print_array(array, n, m);
    return changes;
}
