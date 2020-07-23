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

void print_array(int **array, int n, int m) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%s ", array[i][j] ? "\u2B1B" : "\u2B1C");
        }
        printf("\n");
    }
}

void initialize(int **array, int n, int m) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            array[i][j] = rand() % 2;
        }
    }
//    array[4][1] = 1;
//    array[3][2] = 1;
//    array[3][3] = 1;
//    array[4][3] = 1;
//    array[5][3] = 1;
    print_array(array, n, m);
}

void copy(int **target, int **source, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            target[i][j] = source[i][j];
        }
    }
}

int operate(int **array, int n, int m) {
    // int** old_state = array;
    int left = 0, right = 0;
    int up = 0, down = 0;
    int up_left = 0, up_right = 0;
    int down_left = 0, down_right = 0;
    int **old_array = create(n, m);
    copy(old_array, array, n, m);

    int changes = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            /*
              array[i+1][j], array[i][j+1], array[i-1][j], array[i][j-1],
              array[i+1][j-1], array[i+1][j+1], array[i-1][j+1], array[i-1][j-1]
            */

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

//            printf("current: [%d,%d]\n, left: %d, right: %d\n, up_left: %d, up_right: %d\n, down_left: %d, down_right: %d\n, up: %d, down: %d\n\n\n",
//                   i, j, left, right, up_left, up_right, down_left, down_right, up, down);
        }
    }
    print_array(array, n, m);
    return changes;
}