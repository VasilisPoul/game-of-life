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

/*RESET COLOR*/
#define RESET  "\x1B[0m"

#define N 32
#define M 4
#define FILE_NAME "/home/msi/projects/CLionProjects/game-of-life/cuda/test-files/32x32.txt"
#define STEPS 1000

char **allocate2DArray(int rows, int columns) {
    char **block;
    int i;
    block = (char **) malloc(rows * sizeof(char *));
    block[0] = (char *) malloc(rows * columns * sizeof(char));
    for (i = 1; i < rows; i++) {
        block[i] = &(block[0][i * rows]);
    }
    memset(block[0], '0', rows * columns * sizeof(char));
    return block;
}

void free2DArray(char **block) {
    free(block[0]);
    free(block);
}

void print_array(char **array, bool split, int rowDim, int colDim, int localRowDim, int localColDim) {
    printf("\n");
    for (int i = 0; i < rowDim; i++) {
        for (int j = 0; j < colDim; j++) {
            printf("%s %c ", array[i][j] == '1' ? RED"\u2B1B" RESET : "\u2B1C",
                   (split && (j + 1) % localColDim == 0) ? ' ' : '\0');
        }
        printf("\n%c", (split && (i + 1) % localRowDim == 0) ? '\n' : '\0');
    }
    printf("\n");
}

// Device code
__global__ void kernel(const char *old, char *current) {
    __shared__ char local[M + 2][M + 2];
    int sum = 0;
    unsigned int local_row = threadIdx.x;
    unsigned int local_col = threadIdx.y;
    unsigned int local_thread_id = local_col + local_row * M;

    unsigned int ix = blockIdx.x * (blockDim.x) + threadIdx.x;
    unsigned int iy = blockIdx.y * (blockDim.y) + threadIdx.y;
    unsigned int idx = ix * N + iy;

    // Initialize 'local' shared array
    local[local_row + 1][local_col + 1] = old[idx];

    // Initialize neighbors
    if (blockIdx.x > 0 && blockIdx.x < gridDim.x - 1 && blockIdx.y > 0 && blockIdx.y < gridDim.y - 1) {
        //up
        if (local_row == 0) {
            local[local_row][local_col + 1] = old[idx - N];
        }
        //down
        if (local_row == blockDim.x - 1) {
            local[local_row + 2][local_col + 1] = old[idx + N];
        }
        //left
        if (local_col == 0) {
            local[local_row + 1][local_col] = old[idx - 1];
        }
        //right
        if (local_col == blockDim.y - 1) {
            local[local_row + 1][local_col + 2] = old[idx + 1];
        }
        //up left
        if (local_col == 0 && local_row == 0) {
            local[local_row][local_col] = old[idx - N - 1];
        }
        //up right
        if (local_col == blockDim.y - 1 && local_row == 0) {
            local[local_row][local_col + 2] = old[idx - N + 1];
        }
        //down left
        if (local_col == 0 && local_row == blockDim.y - 1) {
            local[local_row + 2][local_col] = old[idx + N - 1];
        }
        //down right
        if (local_col == blockDim.y - 1 && local_row == blockDim.x - 1) {
            local[local_row + 2][local_col + 2] = old[idx + N + 1];
        }
    } else {
        if (blockIdx.x == 0) {
            //up
            if (local_row == 0) {
                local[local_row][local_col + 1] = old[idx + (N - 1) * N];
            }
            //down
            if (local_row == blockDim.x - 1) {
                local[local_row + 2][local_col + 1] = old[idx + N];
            }
            //left
            if (local_col == 0) {
                if (blockIdx.y == 0) {
                    local[local_row + 1][local_col] = old[idx + N - 1];
                } else {
                    local[local_row + 1][local_col] = old[idx - 1];
                }
            }
            //right
            if (local_col == blockDim.y - 1) {
                if (blockIdx.y != gridDim.y - 1) {
                    local[local_row + 1][local_col + 2] = old[idx + 1];
                } else {
                    local[local_row + 1][local_col + 2] = old[idx - N + 1];
                }
            }
            //up left
            if (local_col == 0 && local_row == 0) {
                if (blockIdx.y == 0) {
                    local[local_row][local_col] = old[idx + N * N - 1];
                } else {
                    local[local_row][local_col] = old[idx + (N - 1) * N - 1];
                }
            }
            //up right
            if (local_row == 0 && local_col == blockDim.y - 1) {
                if (blockIdx.y != gridDim.y - 1) {
                    local[local_row][local_col + 2] = old[idx + (N - 1) * N + 1];
                } else {
                    local[local_row][local_col + 2] = old[idx + (N - 1) * N - N + 1];
                }
            }
            //down left
            if (local_row == blockDim.x - 1 && local_col == 0) {
                if (blockIdx.y == 0) {
                    local[local_row + 2][local_col] = old[idx + 2 * N - 1];
                } else {
                    local[local_row + 2][local_col] = old[idx + 2 * N - 1 - N];
                }
            }
            //down right
            if (local_row == blockDim.x - 1 && local_col == blockDim.y - 1) {
                if (blockIdx.y != gridDim.y - 1) {
                    local[local_row + 2][local_col + 2] = old[idx + 1 * N + 1];
                } else {
                    local[local_row + 2][local_col + 2] = old[idx + N + 1];
                }
            }
        }

        if (blockIdx.x == gridDim.x - 1) {
            //up
            if (local_row == 0) {
                local[local_row][local_col + 1] = old[idx - N];
            }
            //down
            if (local_row == blockDim.x - 1) {
                local[local_row + 2][local_col + 1] = old[idx - N * (N - 1)];
            }
            //left
            if (local_col == 0) {
                if (blockIdx.y == 0) {
                    local[local_row + 1][local_col] = old[idx + N - 1];
                } else {
                    local[local_row + 1][local_col] = old[idx - 1];
                }
            }
            //right
            if (local_col == blockDim.y - 1) {
                if (blockIdx.y != gridDim.y - 1) {
                    local[local_row + 1][local_col + 2] = old[idx + 1];
                } else {
                    local[local_row + 1][local_col + 2] = old[idx - N + 1];
                }
            }
            //up left
            if (local_col == 0 && local_row == 0) {
                if (blockIdx.y == 0) {
                    local[local_row][local_col] = old[idx - 1];
                } else {
                    local[local_row][local_col] = old[idx - N - 1];
                }
            }
            //up right
            if (local_row == 0 && local_col == blockDim.y - 1) {
                if (blockIdx.y != gridDim.y - 1) {
                    local[local_row][local_col + 2] = old[idx - N + 1];
                } else {
                    local[local_row][local_col + 2] = old[idx - 2 * N + 1];
                }
            }
            //down left
            if (local_row == blockDim.x - 1 && local_col == 0) {
                if (blockIdx.y == 0) {
                    local[local_row + 2][local_col] = old[idx - (N - 1) * (N - 1)];
                } else {
                    local[local_row + 2][local_col] = old[idx - N * (N - 1) - 1];
                }
            }
            //down right
            if (local_row == blockDim.x - 1 && local_col == blockDim.y - 1) {
                if (blockIdx.y != gridDim.y - 1) {
                    local[local_row + 2][local_col + 2] = old[idx - (N - 1) * N + 1];
                } else if (blockIdx.y == gridDim.y - 1) {
                    local[local_row + 2][local_col + 2] = old[idx - (N - 1) * N + 1 - N];
                }
            }
        }
        if (blockIdx.x > 0 && blockIdx.x < gridDim.x - 1 && blockIdx.y == 0) {
            //up
            if (local_row == 0) {
                local[local_row][local_col + 1] = old[idx - N];
            }
            //down
            if (local_row == blockDim.x - 1) {
                local[local_row + 2][local_col + 1] = old[idx + N];
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
            if (local_col == blockDim.y - 1 && local_row == 0) {
                local[local_row][local_col + 2] = old[idx - N + 1];
            }
            //down right
            if (local_col == blockDim.y - 1 && local_row == blockDim.x - 1) {
                local[local_row + 2][local_col + 2] = old[idx + N + 1];
            }
            //up left
            if (local_col == 0 && local_row == 0) {
                if (blockIdx.y == 0) {
                    local[local_row][local_col] = old[idx - 1];
                }
            }
            //down left
            if (local_row == blockDim.x - 1 && local_col == 0) {
                if (blockIdx.y == 0) {
                    local[local_row + 2][local_col] = old[idx + 2 * N - 1];
                }
            }
        }
        if (blockIdx.x > 0 && blockIdx.x < gridDim.x - 1 && blockIdx.y == gridDim.y - 1) {
            //up
            if (local_row == 0) {
                local[local_row][local_col + 1] = old[idx - N];
            }
            //down
            if (local_row == blockDim.x - 1) {
                local[local_row + 2][local_col + 1] = old[idx + N];
            }
            //left
            if (local_col == 0) {
                local[local_row + 1][local_col] = old[idx - 1];
            }
            //up left
            if (local_col == 0 && local_row == 0) {
                local[local_row][local_col] = old[idx - N - 1];
            }
            //down left
            if (local_col == 0 && local_row == blockDim.y - 1) {
                local[local_row + 2][local_col] = old[idx + N - 1];
            }
            //right
            if (local_col == blockDim.y - 1) {
                local[local_row + 1][local_col + 2] = old[idx - N + 1];
            }
            //up right
            if (local_row == 0 && local_col == blockDim.y - 1) {
                local[local_row][local_col + 2] = old[idx - 2 * N + 1];
            }
            //down right
            if (local_row == blockDim.x - 1 && local_col == blockDim.y - 1) {
                local[local_row + 2][local_col + 2] = old[idx + 1];
            }
        }
    }

    __syncthreads();

    // Calculate cells
    sum = (local[local_row][local_col] - '0') +
          (local[local_row][local_col + 1] - '0') +
          (local[local_row][local_col + 2] - '0') +
          (local[local_row + 1][local_col] - '0') +
          (local[local_row + 1][local_col + 2] - '0') +
          (local[local_row + 2][local_col] - '0') +
          (local[local_row + 2][local_col + 1] - '0') +
          (local[local_row + 2][local_col + 2] - '0');

    // Is alive
    if ((local[local_row + 1][local_col + 1]) == '1') {
        if (sum <= 1 || sum >= 4) {
            current[idx] = '0';
        } else {
            current[idx] = '1';
        }
    } else if (sum == 3) {
        current[idx] = '1';
    } else {
        current[idx] = '0';
    }
}

// Host code
int main() {
    char **host_array = nullptr, *device_old = nullptr, *device_current = nullptr, *temp = nullptr;;
    int i = 0, fd = 0;
    double time_spent = 0.0;
    clock_t begin, end;

    // Threads (2D) per block
    dim3 m(M, M);

    // Blocks (2D grid)
    dim3 n((unsigned int) ((N + (float) M - 1) / (float) M), (unsigned int) ((N + (float) M - 1) / (float) M));

    assert(N * N == M * M * (n.x * n.y));

    // Array allocations
    host_array = allocate2DArray(N, N);

//    // Read file
    if ((fd = open(FILE_NAME, O_RDONLY)) < 0) {
        fprintf(stderr, "Could not open file \"%s\"\n", FILE_NAME);
        return -1;
    }
    i = 0;
    while (read(fd, &host_array[i++][0], N));
    close(fd);

    printf("host_array before:\n");
    print_array(host_array, true, N, N, N, N);

    // Allocate 2D 'old' array on device
    cudaMalloc((void **) &device_old, N * N * sizeof(char));

    // Copy 2D 'old' array on device
    cudaMemcpy(device_old, host_array[0], N * N * sizeof(char), cudaMemcpyHostToDevice);

    // Allocate 2D 'current' array on device
    cudaMalloc((void **) &device_current, N * N * sizeof(char));

    // Initialize 2D 'current' array on device
    cudaMemset(device_current, '0', N * N * sizeof(char));

    begin = clock();

    // Computations
    for (i = 0; i < STEPS; i++) {
        // Call device function
        kernel<<<n, m>>>(device_old, device_current);

        // Copy 2D 'device_current' array from device to host
        cudaMemcpy(host_array[0], device_current, sizeof(char) * N * N, cudaMemcpyDeviceToHost);

        cudaMemset(device_old, '0', N * N * sizeof(char));

        printf("host_array on step %d:\n", i);
        print_array(host_array, true, N, N, N, N);

        cudaDeviceSynchronize();

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
