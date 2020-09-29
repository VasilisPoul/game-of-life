#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define N 8
#define M 4
#define FILE_NAME "./test-files/8x8.txt"
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
            printf("%s ", array[i][j] == '1' ? "\u2B1C" : "\u2B1B");
        }
        printf("\n");
    }
    printf("\n");
}


// Device code
__global__ void kernel(char *device_old) {
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * N + ix;
    __shared__ char local[M][M];

    int i = threadIdx.y;
    int j = threadIdx.x;




    // local[i][j] = old[idx]

   // local[threadIdx.x] = device_old[i];

    printf("%d %d t[%d][%d] - t[%d][%d]=%d\n", blockIdx.x, blockDim.x, threadIdx.x, threadIdx.y, ix, iy, idx);
}

//Host Code
int main() {
    char **host_array = nullptr, **host_current = nullptr, *device_old, *device_current, *temp = nullptr;;
    int i = 0, fd = 0;
    dim3 m(M, M);
    dim3 n((unsigned int) ((N + (float) M - 1) / (float) M), (unsigned int) ((N + (float) M - 1) / (float) M));

    // Array allocations
    host_array = allocate2DArray(N, N);
    print_array(host_array, N);

    // Read file
    fd = open(FILE_NAME, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Could not open file \"%s\"\n", "320file.txt");
        return -1;
    }
    while (read(fd, host_array[i++], N));
    close(fd);

    print_array(host_array, N);

    // Send 2D to the Device
    cudaMalloc((void **) &device_old, N * N * sizeof(char));
    cudaMemcpy(device_old, host_array[0], sizeof(char) * N * N, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &device_current, N * N * sizeof(char));
    cudaMemset(device_current, '0', sizeof(char) * N * N);

    //TODO: check this
    cudaDeviceSynchronize();

    // Computations
    for (i = 0; i < STEPS; i++) {

        //call device function
        // kernel<<<N, N>>>(device_old/*, device_current*/);

        // Swap device_old and device_current
        temp = device_old;
        device_old = device_current;
        device_current = temp;


    }
    cudaMemcpy(host_array[0], device_current, sizeof(char) * N * N, cudaMemcpyDeviceToHost);
    print_array(host_array, N);
    return 0;
}