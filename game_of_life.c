#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int mod(int x, int m) {
    return (x % m + m) % m;
}

int **create(int n, int m) {

    int **array = malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        array[i] = malloc(m * sizeof(int));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array[i][j] = 0;
        }
    };

    return array;
}

void initialize(int **array, int n, int m) {
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array[i][j] = rand() % 2;
        }
    }
}

void print_array(int **array, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%s ", array[i][j] ? "\u2B1C" : "\u2B1B");
        }
        printf("\n");
    }
}

int operate(int **array, int n, int m) {
    // int** old_state = array;
    int left = 0, right = 0;
    int up = 0, down = 0;
    int up_left = 0, up_right = 0;
    int down_left = 0, down_right = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            /*
              array[i+1][j], array[i][j+1], array[i-1][j], array[i][j-1],
              array[i+1][j-1], array[i+1][j+1], array[i-1][j+1], array[i-1][j-1]
            */

            left = array[i][mod(j - 1, m)];
            up_left = array[mod(i - 1, n)][mod(j - 1, m)];
            down_left = array[mod(i + 1, n)][mod(j - 1, m)];
            right = array[i][mod(j + 1, m)];
            up_right = array[mod(i - 1, n)][mod(j + 1, m)];
            down_right = array[mod(i + 1, n)][mod(j + 1, m)];
            up = array[mod(i - 1, n)][j];
            down = array[mod(i + 1, n)][j];
            int sum = left + right + up_left + up_right + down_left + down_right + up + down;

            printf("current: [%d,%d]\n, left: %d, right: %d\n, up_left: %d, up_right: %d\n, down_left: %d, down_right: %d\n, up: %d, down: %d\n\n\n",
                   i, j, left, right, up_left, up_right, down_left, down_right, up, down);


        }
    }
    return 0;
}