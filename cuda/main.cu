#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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

void print_array(char **array, int n) {
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
//            printf("%s ", array[i][j] == '1' ? "\u2B1B" : "\u2B1C");
            printf("%c ", array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


// Device code
__global__ void kernel(const char *old, char *current) {
    __shared__ char local[M + 2][M + 2];

    unsigned int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_thread_id = global_col + global_row * (N);

    unsigned int local_row = threadIdx.x;
    unsigned int local_col = threadIdx.y;
    unsigned int local_thread_id = local_col + local_row * M;

    unsigned int index = global_thread_id + 2 * local_row - 2 * blockIdx.y - N * blockIdx.x;

    //TODO: fill halo

    local[local_row][local_col] = old[index];

    __syncthreads();

    if (blockIdx.x == 0 && blockIdx.y == 0) {
        if (global_row == 0 && global_col == 0) {

            printf("BC[%d][%d] - TGC[%d][%d]=%d - local:%c\n",
                   blockIdx.x, blockIdx.y,
                   global_row, global_col, global_thread_id,
                   local[local_row][local_col]
            );
            for (int i = 0; i < M + 2; i++) {
                for (int j = 0; j < M + 2; j++) {
                    printf("%c ", local[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    //printf("BC[%d][%d] - TGC[%d][%d]=%d\n", blockIdx.x, blockIdx.y, global_row, global_col, global_thread_id);
}

//Host Code
int main() {
    char **host_array = nullptr, **host_current = nullptr, *device_old, *device_current, *temp = nullptr;;
    int i = 0, fd = 0;
    dim3 m(M + 2, M + 2);
    dim3 n((unsigned int) ((N + 2 + (float) M - 1) / (float) M), (unsigned int) ((N + 2 + (float) M - 1) / (float) M));

    // Array allocations
    host_array = allocate2DArray(N + 2, N + 2);
    print_array(host_array, N + 2);

    // Read file
    fd = open(FILE_NAME, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Could not open file \"%s\"\n", FILE_NAME);
        return -1;
    }

    //host_array+=N+1;
    i = 1;
    while (read(fd, &host_array[i][1], N)) {
        i++;
    }
    close(fd);

    int j = 0;
    for (i = 0; i < N + 2; i++) {
        for (j = 0; j < N + 2; j++) {
            if (i == 0 || j == 0 || i == N + 1 || j == N + 1) {
                host_array[i][j] = '*';
            }
        }
    }

    // Send 2D to the Device
    cudaMalloc((void **) &device_old, (N + 2) * (N + 2) * sizeof(char));
    cudaMemcpy(device_old, host_array[0], sizeof(char) * (N + 2) * (N + 2), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &device_current, (N + 2) * (N + 2) * sizeof(char));
    cudaMemset(device_current, '0', sizeof(char) * (N + 2) * (N + 2));

    //TODO: check this
    //cudaDeviceSynchronize();

    printf("host_array before:\n");
    print_array(host_array, N + 2);

    // Computations
    for (i = 0; i < STEPS; i++) {

        //call device function
        kernel<<<n, m>>>(device_old, device_current);

        cudaMemcpy(host_array[0], device_current, sizeof(char) * (N + 2) * (N + 2), cudaMemcpyDeviceToHost);

        printf("host_array on step %d:\n", i);
        print_array(host_array, (N + 2));

        // Swap device_old and device_current
        temp = device_old;
        device_old = device_current;
        device_current = temp;
    }
    return 0;
}
