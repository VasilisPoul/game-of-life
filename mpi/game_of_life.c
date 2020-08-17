#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include "game_of_life.h"

/*COLOR*/
#define GRAY  "\x1B[90m"
#define RED  "\x1B[31m"
#define GREEN  "\x1B[32m"

/*BOLD-UNDERLINE-COLOR*/
#define B_U_GRAY  "\x1B[1m\x1B[4m\x1B[90m"
#define B_U_RED  "\x1B[1m\x1B[4m\x1B[31m"
#define B_U_GREEN  "\x1B[1m\x1B[4m\x1B[32m"

/*RESET COLOR*/
#define RESET  "\x1B[0m"


bool **allocate2DArray(int rows, int columns) {
    int c;
    bool **array;
    array = (bool **) malloc(rows * sizeof(bool *));
    if (!array)
        return (NULL);
    for (c = 0; c < rows; c++) {
        array[c] = (bool *) malloc(columns * sizeof(bool));
        if (!array[c])
            return (NULL);
    }
    return array;
}

void Free2DArray(bool **array, int rows) {
    int c;
    for (c = 0; c < rows; c++)
        free((bool *) array[c]);
    free((bool **) array);
}

int mod(int x, int m) {
    return (x % m + m) % m;
}

void print_array(bool **array, int n, int m, int dimN, int dimM) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%s %c", array[i][j] == true ? "\u2B1B" : "\u2B1C", ((j + 1) % dimM == 0) ? '\t' : '\0');
        }
        printf("\n%c", ((i + 1) % dimN == 0) ? '\n' : '\0');
    }
    printf("\n");
}

void initialize_array(bool **array, int n, int m) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array[i][j] = (bool) (rand() % 2);
        }
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
    Free2DArray(old_array, TABLE_N);

    return changes;
}
