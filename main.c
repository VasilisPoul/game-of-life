#include <stdio.h>

#include "serial/game_of_life.h"

int main(int argc, char **argv) {
    int array[N][M];
    print_array(array, N, M);
    initialize(array, N, M);
    while (operate(array, N, M));
}
