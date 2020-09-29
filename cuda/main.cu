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
__global__ void kernel(char *old, char* current) {
    __shared__ char local[M][M];
    unsigned int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int local_row = threadIdx.x;
    unsigned int local_col = threadIdx.y;
    unsigned int global_thread_id = global_col * (N+2) + global_row;
    unsigned int local_thread_id = local_col + local_row  * M;

    local[local_row][local_col] = old[global_thread_id];


    printf("BLOCK:[%d][%d] Thread global coords:[%d][%d]=%d Thread local coords:[%d][%d]=%d\n", blockIdx.x, blockIdx.y, global_row, global_col, global_thread_id, local_row, local_col, local_thread_id   );
}

//Host Code
int main() {
    char **host_array = nullptr, **host_current = nullptr, *device_old, *device_current, *temp = nullptr;;
    int i = 0, fd = 0;
    dim3 m(M, M);
    dim3 n((unsigned int) ((N+2 + (float) M - 1) / (float) M), (unsigned int) ((N+2 + (float) M - 1) / (float) M));

    // Array allocations
    host_array = allocate2DArray(N+2, N+2);
    print_array(host_array, N+2);

    // Read file
    fd = open(FILE_NAME, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Could not open file \"%s\"\n", "320file.txt");
        return -1;
    }

    //host_array+=N+1;
    i=1;
    while (read(fd, &host_array[i][1], N)){
        i++;
    }
    close(fd);

    // Send 2D to the Device
    cudaMalloc((void **) &device_old, (N+2) * (N+2) * sizeof(char));
    cudaMemcpy(device_old, host_array[0], sizeof(char) * (N+2) * (N+2), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &device_current, (N+2) * (N+2) * sizeof(char));
    cudaMemset(device_current, '0', sizeof(char) * (N+2) * (N+2));

    //TODO: check this
    //cudaDeviceSynchronize();

    printf("host_array before:\n");
    print_array(host_array, N+2);

    // Computations
    for (i = 0; i < STEPS; i++) {

        //call device function
        kernel<<<n, m>>>(device_old, device_current);

        cudaMemcpy(host_array[0], device_current, sizeof(char) * (N+2) * (N+2), cudaMemcpyDeviceToHost);
       
        printf("host_array on step %d:\n", i);
        print_array(host_array, (N+2));

        // Swap device_old and device_current
        temp = device_old;
        device_old = device_current;
        device_current = temp;
    }
    return 0;
}
