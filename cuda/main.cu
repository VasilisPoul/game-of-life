#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "game_of_life.h"

int main(int argc, char **argv) {
    char **array;
    array = allocate2DArray(N, M);
    print_array(array, N, M);
    initialize(array, N, M);


    cudaAllocate2DArray<<<1, rc_block_size>>>(dev_array1);


//    while (operate(array, N, M));
//
//    Free2DArray(array, N);
}
