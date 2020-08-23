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


char **allocate2DArray(int rows, int columns) {
    char **array;
    int i = 0;
    array = malloc(rows * sizeof(char *));
    array[0] = malloc(rows * columns * sizeof(char));
    for (i = 1; i < rows; i++) {
        array[i] = &(array[0][i * rows]);
    }
    return array;
}

void free2DArray(char **array, int rows) {
    free(array[0]);
    free(array);
}

int mod(int x, int m) {
    return (x % m + m) % m;
}

void print_array(char **array, bool split, bool internals, int rowDim, int colDim, int localRowDim, int localColDim) {
    printf("\n");
    for (int i = 0; i < rowDim; i++) {
        for (int j = 0; j < colDim; j++) {
            if ((rowDim != localRowDim && colDim != localColDim)) {
                printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B"RESET : "\u2B1C",
                       (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
            } else {
                if ((i == 0 || i == rowDim - 1) || (j == 0 || j == colDim - 1)) {
                    printf("%s %c ", array[i][j] == '1' ? B_GREEN"\u2B1B"RESET : "\u2B1C",
                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                } else if (internals && ((i == 1 || i == rowDim - 2) || (j == 1 || j == colDim - 2))) {
                    printf("%s %c ", array[i][j] == '1' ? BLUE"\u2B1B"RESET : "\u2B1C",
                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                } else {
                    printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B"RESET : "\u2B1C",
                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                }
            }
        }
        printf("\n%c", (split && (i + 1) % localRowDim == 0) ? '\n' : '\0');
    }
    printf("\n");
}


void initialize_block(char **block, bool zeroFill, int n, int m) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            block[i][j] = zeroFill ? '0' : rand() % 2 == 0 ? '0' : '1';
        }
    }
}

// Inline calculate
inline void calculate(char **old, char **current, int i, int j, int *changes) {
    int sum = (old[i - 1][j - 1] - 48) +
              (old[i - 1][j] - 48) +
              (old[i - 1][j + 1] - 48) +
              (old[i][j - 1] - 48) +
              (old[i][j + 1] - 48) +
              (old[i + 1][j - 1] - 48) +
              (old[i + 1][j] - 48) +
              (old[i + 1][j + 1] - 48);

    // Is alive
    if ((old[i][j]) == '1') {
        if (sum <= 1 || sum >= 4) {
            current[i][j] = '0';
            (*changes)++;
        } else {
            current[i][j] = '1';
        }
    } else if (sum == 3) {
        current[i][j] = '1';
        (*changes)++;
    } else {
        current[i][j] = '0';
    }
}
