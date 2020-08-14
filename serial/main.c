#include <stdio.h>
#include <stdbool.h>
#include "game_of_life.h"

int main(int argc, char **argv) {
    bool **array;
    array = allocate2DArray(N, M);
    print_array(array, N, M);
    initialize(array, N, M);

    while (operate(array, N, M));

    Free2DArray(array, N);
}
