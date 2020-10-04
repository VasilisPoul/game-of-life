#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

/*COLOR*/
#define RED "\x1B[31m"
#define BLUE "\x1B[34m"

/*BOLD-COLOR*/
#define B_RED "\x1B[1m\x1B[31m"
#define B_GREEN "\x1B[1m\x1B[32m"

/*RESET COLOR*/
#define RESET  "\x1B[0m"

#define N 16
#define M 4
#define FILE_NAME "/home/msi/projects/CLionProjects/game-of-life/cuda/test-files/64x64.txt"
#define STEPS 1

int **allocate2DArray(int rows, int columns) {
    int **block;
    int i;
    block = (int **) malloc(rows * sizeof(int *));
    block[0] = (int *) malloc(rows * columns * sizeof(int));
    for (i = 1; i < rows; i++) {
        block[i] = &(block[0][i * rows]);
    }
    memset(block[0], 0, rows * columns * sizeof(int));
    return block;
}

void free2DArray(int **block) {
    free(block[0]);
    free(block);
}

void print_array(int **array, bool split, bool internals, int rowDim, int colDim, int localRowDim, int localColDim) {
    printf("\n");
    for (int i = 0; i < rowDim; i++) {
        for (int j = 0; j < colDim; j++) {
            if ((rowDim != localRowDim && colDim != localColDim)) {
//                printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B" RESET : "\u2B1C",
//                       (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                if (array[i][j]) {
                    printf(RED"%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                } else {
                    printf("%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                }
            } else {
                if ((i == 0 || i == rowDim - 1) || (j == 0 || j == colDim - 1)) {
//                    printf("%s %c ", array[i][j] == '1' ? B_GREEN"\u2B1B" RESET : "\u2B1C",
//                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    if (array[i][j]) {
                        printf(RED"%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    } else {
                        printf("%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    }
                } else if (internals && ((i == 1 || i == rowDim - 2) || (j == 1 || j == colDim - 2))) {
//                    printf("%s %c ", array[i][j] == '1' ? BLUE"\u2B1B" RESET : "\u2B1C",
//                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    if (array[i][j]) {
                        printf(RED"%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    } else {
                        printf("%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    }
                } else {
//                    printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B" RESET : "\u2B1C",
//                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    if (array[i][j]) {
                        printf(RED"%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    } else {
                        printf("%5.4d%c" RESET, array[i][j], (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    }
                }
            }
        }
        printf("\n%c", (split && (i + 1) % localRowDim == 0) ? '\n' : '\0');
    }
    printf("\n");
}

// Device code
__global__ void kernel(int *old, int *current) {
    __shared__ int local[M + 2][M + 2];
    unsigned int local_row = threadIdx.x;
    unsigned int local_col = threadIdx.y;
    unsigned int local_thread_id = local_col + local_row * M;

    unsigned int ix = blockIdx.x * (blockDim.x) + threadIdx.x;
    unsigned int iy = blockIdx.y * (blockDim.y) + threadIdx.y;
    unsigned int idx = ix * N + iy;

    // // split internals
    // if (blockIdx.x > 0 && blockIdx.x < gridDim.x - 1 && blockIdx.y > 0 && blockIdx.y < gridDim.y - 1) {
    //     local[local_row + 1][local_col + 1] = old[idx];

    //     //up
    //     if (local_row == 0) {
    //         local[local_row][local_col + 1] = old[idx - N];
    //     }

    //     //down
    //     if (local_row == blockDim.x - 2) {
    //         local[local_row + 3][local_col + 1] = old[idx + 2 * N];
    //     }

    //     //left
    //     if (local_col == 0) {
    //         local[local_row + 1][local_col] = old[idx - 1];
    //     }

    //     //right
    //     if (local_col == blockDim.y - 2) {
    //         local[local_row + 1][local_col + 3] = old[idx + 2];
    //     }

    //     //up left
    //     if (local_col == 0 && local_row == 0) {
    //         local[local_row][local_col] = old[idx - N - 1];
    //     }

    //     //up right
    //     if (local_col == blockDim.y - 2 && local_row == 0) {
    //         local[local_row][local_col + 3] = old[idx - N + 2];
    //     }

    //     //down left
    //     if (local_col == 0 && local_row == blockDim.y - 2) {
    //         local[local_row + 3][local_col] = old[idx + 2 * N - 1];
    //     }

    //     //down right
    //     if (local_col == blockDim.y - 2 && local_row == blockDim.x - 2) {
    //         local[local_row + 3][local_col + 3] = old[idx + 2 * N + 2];
    //     }
    // } else {
        // // Todo: calculate external blocks


        if(blockIdx.x == 0){
            // //internals
            // local[local_row + 1][local_col + 1] = old[idx];

            // //up
            // if (local_row == 0){
            //     local[local_row][local_col + 1] = old[idx + (N-1) * N];
            // }
            // //down idio me ta apo panw 
            // if (local_row == blockDim.x - 2){
            //     local[local_row + 3][local_col + 1] = old[idx + 2 * N];
            // }
            // //left
            // if (local_col == 0){
            //     //full aristera
            //     if (blockIdx.y == 0){
            //         local[local_row + 1][local_col] = old[idx + N - 1];
            //     }
            //     else{
            //         local[local_row + 1][local_col] = old[idx - 1];
            //     }
            // }
            // //right
            // if(local_col == blockDim.y-1){
                
            //     if (blockIdx.y != gridDim.y - 1){
                    
            //         local[local_row + 1][local_col+2] = old[idx + 1];  
            //     }
            //     //full deksia
            //     else{
            //         printf("Hey!!!\n");
            //         local[local_row + 1][local_col + 2] = old[idx - N+1];
                    
            //         printf("local_row+1: %d, local_col+2: %d\n", local_row+1, local_col+2);
            //         printf("old[idx + 1]: %5.4d\n", old[idx + 1]); 
            //     }
            // }
            // //up left
            // if(local_col == 0 && local_row == 0){
            //     //terma aristera block
            //     if (blockIdx.y == 0){
            //         local[local_row][local_col] = old[idx + N * N - 1];
            //     }
            //     else{
                    
            //         local[local_row][local_col] = old[idx + (N-1)*N - 1];
            //     }
            // }
            // //up right
            // if (local_row == 0 && local_col == blockDim.y - 2){
            //     if(blockIdx.y != gridDim.y - 1){
            //         local[local_row][local_col + 3] = old[idx + (N-1)*N + 2]; 
            //     }
            //     //terma deksia
            //     else{
            //         local[local_row][local_col + 3] = old[idx + (N-1)*N- N+2 ];
            //     }
            // }
            // //down left
            // if(local_row == blockDim.x - 2 && local_col == 0){
            //     //terma aristera block
            //     if (blockIdx.y == 0){
            //         local[local_row+3][local_col] = old[idx + 3 * N - 1];
            //     }
            //     else {
            //         local[local_row+3][local_col] = old[idx + 3 * N - 1 - N];
            //     }
            // }
            // //down right
            // if (local_row == blockDim.x-2 && local_col == blockDim.y - 2){
            //     if(blockIdx.y != gridDim.y - 1){
            //         local[local_row + 3][local_col + 3] = old[idx + 2*N +2];   
            //     }
            //     //terma deksia
            //     else {
            //         //TODO
            //         local[local_row + 3][local_col + 3] = old[idx + N +2];
            //         // printf("local_row+3: %d, local_col+3: %d\n", local_row+3, local_col+3);
            //         // printf("old[idx + N +2]: %5.4d\n", old[idx + N +2]); 
            //     }
            // }            
        }


        if (blockIdx.x == gridDim.x - 1 ) {
            // local[local_row + 1][local_col + 1] = old[idx];

            // //up
            // if (local_row == 0){
            //     //idio me apo panw
            //     local[local_row][local_col + 1] = old[idx - N];
            // }
            // //down idio me ta apo panw 
            // if (local_row == blockDim.x - 1){
            //     local[local_row + 2][local_col + 1] = old[idx - N*(N-1)]; 
            // }
            // //left
            // if (local_col == 0){
            //     //full aristera
            //     if (blockIdx.y == 0){
            //         local[local_row + 1][local_col] = old[idx + N - 1];
            //         // printf("local_row+1: %d, local_col: %d\n", local_row+1, local_col);
            //         // printf("old[idx + N - 1]: %5.4d\n", old[idx + N - 1]); 
            //     }
            //     else{
            //         local[local_row + 1][local_col] = old[idx - 1];
            //     }
            // }
            // //right
            // if(local_col == blockDim.y-1){
                
            //     if (blockIdx.y != gridDim.y - 1){
                    
            //         local[local_row + 1][local_col+2] = old[idx + 1];  
            //     }
            //     //full deksia
            //     else{
            //         local[local_row + 1][local_col + 2] = old[idx - N+1];
                     
            //     }
            // }
            // //up left
            // if(local_col == 0 && local_row == 0){
            //     //terma aristera block
            //     if (blockIdx.y == 0){
            //         local[local_row][local_col] = old[idx -1];
            //     }
            //     else{
            //         local[local_row][local_col] = old[idx - N -1];
            //     }
            // }
            // //up right
            // if (local_row == 0 && local_col == blockDim.y - 1){
            //     if(blockIdx.y != gridDim.y - 1){
            //         local[local_row][local_col + 2] = old[idx - N + 1]; 

            //     }
            //     //terma deksia
            //     else{
            //         //TODO
            //         local[local_row][local_col + 2] = old[idx- 2*N +1];
            //     }
            // }
            // //down left
            // if(local_row == blockDim.x - 1 && local_col == 0){
            //     //terma aristera block
            //     if (blockIdx.y == 0){
            //         local[local_row+2][local_col] = old[idx - (N-1) * (N-1) ];  
            //     }
                
            //     else {
            //         local[local_row+2][local_col] = old[idx - N * (N-1)-1];

            //         printf("local_row+2: %d, local_col: %d\n", local_row+2, local_col);
            //         printf("old[idx-N-1]: %5.4d\n", old[idx - N * (N-1)-1]);
            //     }
            // }
            // //down right
            // if (local_row == blockDim.x-1 && local_col == blockDim.y - 1){
            //     if(blockIdx.y != gridDim.y - 1){
                    
            //         local[local_row+2][local_col+2] = old[idx - (N-1) * N + 1];
            //     }
            //     else if (blockIdx.y == gridDim.y - 1){
            //         local[local_row+2][local_col+2] = old[idx -(N-1)*N + 1 - N];

            //         printf("local_row+2: %d, local_col: %d\n", local_row+2, local_col);
            //         printf("old[idx-N-1]: %5.4d\n", old[idx -(N-1)*N + 1 - N]);
            //     }
               
            // }            


        }

        old[idx] = idx;

        if (blockIdx.x == 1 && blockIdx.y == 0){
            local[local_row + 1][local_col + 1] = old[idx];

                //up
                if (local_row == 0) {
                    local[local_row][local_col + 1] = old[idx - N];
                }

                //down
                if (local_row == blockDim.x - 2) {
                    local[local_row + 3][local_col + 1] = old[idx + 2 * N];
                }

                //right
                if (local_col == blockDim.y - 1) {
                    local[local_row + 1][local_col + 2] = old[idx + 1];
                }
                //left
                if (local_col == 0) {
                    local[local_row + 1][local_col] = old[idx + N - 1];
                }
                //up right
                if (local_col == blockDim.y - 2 && local_row == 0) {
                    local[local_row][local_col + 3] = old[idx - N + 2];
                }
                //down right
                if (local_col == blockDim.y - 2 && local_row == blockDim.x - 2) {
                    local[local_row + 3][local_col + 3] = old[idx + 2 * N + 2];
                }
                //up left
                if(local_col == 0 && local_row == 0){
                    //terma aristera block
                    if (blockIdx.y == 0){
                        local[local_row][local_col] = old[idx -1];
                    }
                
                }

                //down left
                if(local_row == blockDim.x - 2 && local_col == 0){
                    //terma aristera block
                    if (blockIdx.y == 0){
                        local[local_row+3][local_col] = old[idx + 3 * N - 1];
                    }
                }
                


        }
        

        __syncthreads();
       //print block
       if (ix == 4 && iy == 0) {
           for (int i = 0; i < M + 2; i++) {
               for (int j = 0; j < M + 2; j++) {
                    if (i > 0 && i < M+1 && j > 0 && j < M+1){
                        printf(RED"%5.4d " RESET, local[i][j]);
                    }else{
                        printf("%5.4d ", local[i][j]);
                    }
                 
               }
               printf("\n");
           }
           printf("\n");
       }


//    }


    __syncthreads();

    //Todo: Calculate cells

}

// Host code
int main() {
    int **host_array = nullptr, *device_old = nullptr, *device_current = nullptr, *temp = nullptr;;
    int i = 0, fd = 0;
    double time_spent = 0.0;
    clock_t begin, end;

    // Threads (2D) per block
    dim3 m(M, M);

    // Blocks (2D grid)
    dim3 n((unsigned int) ((N + (float) M - 1) / (float) M), (unsigned int) ((N + (float) M - 1) / (float) M));

    //assert(N == M * M * 2);

    // Array allocations
    host_array = allocate2DArray(N, N);

//    // Read file
//    if ((fd = open(FILE_NAME, O_RDONLY)) < 0) {
//        fprintf(stderr, "Could not open file \"%s\"\n", FILE_NAME);
//        return -1;
//    }
//    i = 1;
//    while (read(fd, &host_array[i++][1], N));
//    close(fd);

    printf("host_array before:\n");
    print_array(host_array, true, true, N, N, N, N);

    // Initialize 2D 'old' array on device
    cudaMalloc((void **) &device_old, N * N * sizeof(int));

    // Copy 2D 'old' array on device
    cudaMemcpy(device_old, host_array[0], N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize 2D 'current' array on device
    cudaMalloc((void **) &device_current, N * N * sizeof(int));

    // Copy 2D 'current' array on device
    cudaMemset(device_current, '0', N * N * sizeof(int));

    begin = clock();

    // Computations
    for (i = 0; i < STEPS; i++) {
        // Call device function
        kernel<<<n, m>>>(device_old, device_current);

        // Copy 2D 'device_current' array on host
        cudaMemcpy(host_array[0], device_old, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

        printf("host_array on step %d:\n", i);
        print_array(host_array, true, true, N, N, N, N);

        // Swap 'device_old' and 'device_current' arrays
        temp = device_old;
        device_old = device_current;
        device_current = temp;
    }

    end = clock();

    time_spent = (double) (end - begin) / CLOCKS_PER_SEC;

    printf("time_spent=%f\n", time_spent);

    // Free memory
    cudaFree(device_old);
    cudaFree(device_current);
    free2DArray(host_array);
    return 0;
}
