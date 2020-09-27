// #include <stdlib.h>
// #include <stdio.h>
// #include <stdbool.h>
// #include <cuda.h>

// void cudaAllocate2DArray(char **block, int rows, int columns) {
//     int i;
//     printf("test1\n");
//     cudaMalloc((void ***) &block, rows * sizeof(char *));



//     //    block = (char **) malloc(rows * sizeof(char *));
//     cudaMalloc((void **) &block[0], rows * columns * sizeof(char));




//     printf("test2\n");
//     //block[0] = (char *) malloc(rows * columns * sizeof(char));
//     for (i = 1; i < rows; i++) {
//         block[i] = &(block[0][i * rows]);
//     }
//     // memset(block[0], (int) '0', rows * columns * sizeof(char));
//     return;
// }


// int main(int argc, char **argv) {
//     char **block = NULL;
//     char **device_block = NULL;
//     int rows = 960;
//     int columns = 960;
    
//     cudaAllocate2DArray(device_block, rows, columns);
    

//     cudaMemset( device_block, 0, rows*columns*sizeof(char));


// //    block = allocate2DArray(N, M);
// //    print_array(block, N, M);
// //    initialize(block, N, M);

// //    while (operate(array, N, M));
// //
// //    Free2DArray(array, N);
// }




#include<stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define DIMS 320


char **allocate2DArray(int rows, int columns) {
    int c;
    char **array;
    array = (char **) malloc(rows * sizeof(char *));
    if (!array)
        return (NULL);
    for (c = 0; c < rows; c++) {
        array[c] = (char *) malloc(columns * sizeof(char));
        if (!array[c])
            return (NULL);
    }
    return array;
}


// Device code
__global__ void kernel(char* devPtr, int pitch)
{
    for (int r = 0; r < DIMS; ++r) {
        char* row = (char*)   (  devPtr + r * pitch  );
        for (int c = 0; c < DIMS; ++c) {
             char element = row[c];
        }
    }
}

//Host Code
int main() {
    char** host_block = NULL;
    char* devPtr;
    size_t pitch;
    host_block = allocate2DArray(DIMS, DIMS);

    int fd = open("320file.txt", O_RDONLY);
    if(fd < 0){
        fprintf(stderr, "Could not open file \"%s\"\n", "320file.txt");
        return -1;
    }

    printf("1\n");

    int i = 0;
    while(read(fd,  &host_block[i], DIMS)){
        i+=DIMS;
    }
    close(fd);

    printf("2\n");

    printf("\n");
    for(i=0;i<DIMS;i++){
        for(int j=0;j<DIMS;j++){
            printf("%c", host_block[i][j]);
        }
        printf("\n");
    }

    printf("3\n");


    // // arxeio: monodiastato

    // // host: 2d

    // // gpu: 2d
    // cudaMallocPitch((void**)&devPtr, &pitch, DIMS * sizeof(char), DIMS);

    // printf("\npitch=%d\n", pitch);

    // kernel<<<100, 512>>>(devPtr, pitch);
    return 0;
}