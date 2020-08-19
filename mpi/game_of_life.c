#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "game_of_life.h"

/*COLOR*/
#define RED "\x1B[31m"
#define BLUE "\x1B[34m"

/*BOLD-COLOR*/
#define B_RED "\x1B[1m\x1B[31m"
#define B_GREEN "\x1B[1m\x1B[32m"

/*RESET COLOR*/
#define RESET  "\x1B[0m"


bool **allocate2DArray(int rows, int columns) {
    bool **array;
    int i = 0;
    array = malloc(rows * sizeof(bool *));
    array[0] = malloc(rows * columns * sizeof(bool));
    for (i = 1; i < rows; i++) {
        array[i] = &(array[0][i * rows]);
    }
    return array;
}

void free2DArray(bool **array, int rows) {
    free(array[0]);
    free(array);
}

int mod(int x, int m) {
    return (x % m + m) % m;
}

void print_array(bool **array, int rowDim, int colDim, int localRowDim, int localColDim) {
    printf("\n");
    for (int i = 0; i < rowDim; i++) {
        for (int j = 0; j < colDim; j++) {
            if ((rowDim != localRowDim && colDim != localColDim)) {
                printf("%s %c", array[i][j] == true ? RED"\u2B1B"RESET : "\u2B1C",
                       ((j + 1) % localColDim == 0) ? '\t' : '\0');
            } else {
                if ((i == 0 || i == rowDim - 1) || (j == 0 || j == colDim - 1)) {
                    printf("%s %c", array[i][j] == true ? B_GREEN"\u2B1B"RESET : "\u2B1C",
                           ((j + 1) % localColDim == 0) ? '\t' : '\0');
                } else if ((i == 1 || i == rowDim - 2) || (j == 1 || j == colDim - 2)) {
                    printf("%s %c", array[i][j] == true ? BLUE"\u2B1B"RESET : "\u2B1C",
                           ((j + 1) % localColDim == 0) ? '\t' : '\0');
                } else {
                    printf("%s %c", array[i][j] == true ? RED"\u2B1B"RESET : "\u2B1C",
                           ((j + 1) % localColDim == 0) ? '\t' : '\0');
                }
            }
        }
        printf("\n%c", ((i + 1) % localRowDim == 0) ? '\n' : '\0');
    }
    printf("\n");
}


void initialize_array(bool **array, int n, int m) {
    srand(12345/*time(NULL)*/);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array[i][j] = (bool) (rand() % 2);
        }
    }
}

// Inline calculate
inline void calculate(bool **a, bool **b, int i, int j, int *changes) {
    int sum = 0;
    sum = a[i - 1][j - 1] + a[i - 1][j] + a[i - 1][j + 1] + a[i][j - 1] +
          a[i][j + 1] + a[i + 1][j - 1] + a[i + 1][j] + a[i + 1][j + 1];
    if (a[i][j]) {
        if (sum <= 1 || sum >= 4) {
            b[i][j] = false;
            (*changes)++;
        } else {
            b[i][j] = true;
        }
    } else if (sum == 3) {
        b[i][j] = true;
        (*changes)++;
    }
}

void copy(bool **target, bool **source, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            target[i][j] = source[i][j];
        }
    }
}

int operate(bool **array, int n, int m) {
    bool **old_array;
    old_array = allocate2DArray(TABLE_N, TABLE_M);
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
    free2DArray(old_array, TABLE_N);

    return changes;
}
