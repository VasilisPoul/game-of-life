#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

/*COLOR*/
#define RED "\x1B[31m"
#define BLUE "\x1B[34m"

/*BOLD-COLOR*/
#define B_RED "\x1B[1m\x1B[31m"
#define B_GREEN "\x1B[1m\x1B[32m"

/*RESET COLOR*/
#define RESET  "\x1B[0m"

#define N 32
#define M 16
#define FILE_NAME "/home/msi/projects/CLionProjects/game-of-life/cuda/test-files/32x32.txt"
#define STEPS 1

char **allocate2DArray(int rows, int columns) {
    char **block;
    int i;
    block = (char **) malloc(rows * sizeof(char *));
    block[0] = (char *) malloc(rows * columns * sizeof(char));
    for (i = 1; i < rows; i++) {
        block[i] = &(block[0][i * rows]);
    }
    memset(block[0], (int) '0', rows * columns * sizeof(char));
    return block;
}

void free2DArray(char **block) {
    free(block[0]);
    free(block);
}

void print_array(char **array, bool split, bool internals, int rowDim, int colDim, int localRowDim, int localColDim) {
    printf("\n");
    for (int i = 0; i < rowDim; i++) {
        for (int j = 0; j < colDim; j++) {
            if ((rowDim != localRowDim && colDim != localColDim)) {
//                printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B" RESET : "\u2B1C", (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                printf("%c %c ", array[i][j],
                       (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
            } else {
                if ((i == 0 || i == rowDim - 1) || (j == 0 || j == colDim - 1)) {
//                    printf("%s %c ", array[i][j] == '1' ? B_GREEN"\u2B1B" RESET : "\u2B1C",(split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    printf("%c %c ", array[i][j],
                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');

                } else if (internals && ((i == 1 || i == rowDim - 2) || (j == 1 || j == colDim - 2))) {
//                    printf("%s %c ", array[i][j] == '1' ? BLUE"\u2B1B" RESET : "\u2B1C", (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    printf("%c %c ", array[i][j],
                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                } else {
//                    printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B" RESET : "\u2B1C", (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                    printf("%c %c ", array[i][j],
                           (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
                }
            }
        }
        printf("\n%c", (split && (i + 1) % localRowDim == 0) ? '\n' : '\0');
    }
    printf("\n");
}

// Device code
__global__ void kernel(char *old, char *current) {
    __shared__ char local[M + 2][M + 2];

    unsigned int ix = blockIdx.x * (blockDim.x) + threadIdx.x;
    unsigned int iy = blockIdx.y * (blockDim.y) + threadIdx.y;
    unsigned int idx = iy * (N) + ix;

    unsigned int local_row = threadIdx.x;
    unsigned int local_col = threadIdx.y;
    unsigned int local_thread_id = local_col + local_row * M;

    unsigned int index = idx + 2 * local_row - 2 * blockIdx.y - N * blockIdx.x;
    old[idx] = '2';




    /*    //old[index - N] = '0';
    // Up

    if (blockIdx.x == 0) {
        //old[index - N] = old[index + N * (N - 1)];
        //old[index - N] = '1';
    }

    // Down
    if (blockIdx.x == blockDim.x) {

    }

    // Left
    if (blockIdx.y == 0) {

    }

    // Right
    if (blockIdx.y == blockDim.y) {

    }*/
    //local[local_row][local_col] = old[index];
    __syncthreads();

/*    if (blockIdx.x == 0 && blockIdx.y == 0) {
        if (global_row == 0 && iy == 0) {

            printf("BC[%d][%d] - TGC[%d][%d]=%d - local:%c\n",
                   blockIdx.x, blockIdx.y,
                   global_row, iy, global_thread_id,
                   local[local_row][local_col]
            );

//            for (int i = 0; i < M + 2; i++) {
//                for (int j = 0; j < M + 2; j++) {
//                    printf("%c ", local[i][j]);
//                }
//                printf("\n");
//            }
//            printf("\n");

            printf("\n");
            for (int i = 0; i < M + 2; i++) {
                for (int j = 0; j < M + 2; j++) {
                    printf("%s  ", local[i][j] == '1' ? RED"\u2B1B" RESET : "\u2B1C");
                }
                printf("\n");
            }
            printf("\n");


        }
    }*/

//printf("BC[%d][%d] - TGC[%d][%d]=%d\n", blockIdx.x, blockIdx.y, global_row, iy, global_thread_id);
}

//Host Code
int main() {
    char **host_array = nullptr, *device_old = nullptr, *device_current = nullptr, *temp = nullptr;;
    int i = 0, fd = 0;

    dim3 m(M, M);

    dim3 n((unsigned int) ((N + (float) M - 1) / (float) M),
           (unsigned int) ((N + (float) M - 1) / (float) M)
    );

    assert(N != M * m.x);

    // Array allocations
    host_array = allocate2DArray(N + 2, N + 2);

    // Read file
    /*
        if ((fd = open(FILE_NAME, O_RDONLY)) < 0) {
            fprintf(stderr, "Could not open file \"%s\"\n", FILE_NAME);
            return -1;
        }
        i = 1;
        while (read(fd, &host_array[i][1], N)) {
            i++;
        }
        close(fd);*/

    printf("host_array before:\n");
    print_array(host_array, true, true, N + 2, N + 2, N + 2, N + 2);

    // Send 2D to the Device
    cudaMalloc((void **) &device_old, (N + 2) * (N + 2) * sizeof(char));
    cudaMemcpy(device_old, host_array[0], (N + 2) * (N + 2) * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &device_current, (N + 2) * (N + 2) * sizeof(char));
    cudaMemset(device_current, '0', (N + 2) * (N + 2) * sizeof(char));

    // Computations
    for (i = 0; i < STEPS; i++) {
        //todo: fill halo cells on device_old array

        //call device function
        kernel<<<n, m>>>(device_old, device_current);

        cudaMemcpy(host_array[0], device_old, sizeof(char) * (N + 2) * (N + 2), cudaMemcpyDeviceToHost);

        printf("host_array on step %d:\n", i);
        print_array(host_array, true, true, N + 2, N + 2, N + 2, N + 2);

        // Swap device_old and device_current
        temp = device_old;
        device_old = device_current;
        device_current = temp;
    }

    // Free memory
    cudaFree(device_old);
    cudaFree(device_current);
    free2DArray(host_array);
    return 0;
}
