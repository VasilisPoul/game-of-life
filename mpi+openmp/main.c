#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <getopt.h>
#include "mpi.h"
#include "game_of_life.h"
#include <omp.h>

#define STEPS 10000

int main(int argc, char **argv) {
    int opt = 0, s = 1, ss = 0, i = 0, j = 0, rank, size = 0, root = 0, rows = 0, cols = 0, inputFileNotExists = 0, starts[2], stepGlobalChanges = 0, stepLocalChanges = 0, stepLocalThreadChanges = 0, thread_id = 0, threads = 0;
    double start_w_time = 0.0, end_w_time = 0.0, local_time = 0.0, max_time = 0.0;
    char **block = NULL, **old = NULL, **current = NULL, **temp = NULL, buffer[100], *inputFilePath = NULL, *outputFolder = NULL;
    MPI_Datatype colType, rowType, subArrayType;
    MPI_Request send_a_request[8], recv_a_request[8], send_b_request[8], recv_b_request[8];
    MPI_Status send_a_status[8], send_b_status[8], recv_a_status[8], recv_b_status[8];

    MPI_Request *fileRequests;
    MPI_Status *fileStatus;

    MPI_File inputFile, outputFile;
    GridInfo grid;

    MPI_Init(&argc, &argv);

    while ((opt = getopt(argc, argv, "i:f:r:c:")) != -1) {
        switch (opt) {
            case 'i':
                inputFilePath = optarg;
                break;
            case 'f':
                outputFolder = optarg;
                break;
            case 'r':
                rows = atoi(optarg);
                break;
            case 'c':
                cols = atoi(optarg);
                break;
            default: /* '?' */
                fprintf(stderr,
                        "Usage: %s [-i] input file path [-f] output folder [-r] number of rows [-c] number of columns\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (argc != 9) {
        fprintf(stderr,
                "Usage: %s [-i] input file path [-f] output folder [-r] number of rows [-c] number of columns\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setupGrid(&grid, rows, cols);

    // Allocate local blocks
    old = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);
    current = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);

    fileRequests = malloc(grid.localBlockDims[0] * sizeof(MPI_Request));
    fileStatus = malloc(grid.localBlockDims[0] * sizeof(MPI_Status));

    // rowType
    MPI_Type_vector(1, grid.localBlockDims[1], 0, MPI_CHAR, &rowType);
    MPI_Type_commit(&rowType);

    // colType
    MPI_Type_vector(grid.localBlockDims[0], 1, grid.localBlockDims[1] + 2, MPI_CHAR, &colType);
    MPI_Type_commit(&colType);

    // Create subArrayType
    starts[0] = grid.gridCoords[0] * grid.localBlockDims[0];
    starts[1] = grid.gridCoords[1] * grid.localBlockDims[1];
    MPI_Type_create_subarray(2, grid.blockDims, grid.localBlockDims, starts, MPI_ORDER_C, MPI_CHAR, &subArrayType);
    MPI_Type_commit(&subArrayType);

    // Open input file
    inputFileNotExists = MPI_File_open(MPI_COMM_WORLD, inputFilePath, MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);
    if (inputFileNotExists) {
        // No file, generate array
        if (rank == root) {
            block = allocate2DArray(grid.blockDims[0], grid.blockDims[1]);
            initialize_block(block, false, grid.blockDims[0], grid.blockDims[1]);
            printf("block: (memory)\n");
            print_array(block, true, true, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
                        grid.localBlockDims[1]);
        }
        // Scatter block
        scatter2DArray(block, old, root, &grid);
    } else {
        // Read from file
        MPI_File_set_view(inputFile, 0, MPI_CHAR, subArrayType, "native", MPI_INFO_NULL);
        for (i = 1; i <= grid.localBlockDims[0]; i++) {
            MPI_File_iread(inputFile, &old[i][1], grid.localBlockDims[1], MPI_CHAR, &fileRequests[i - 1]);
        }
        // Wait until reading is done
        MPI_Waitall(grid.localBlockDims[0], fileRequests, fileStatus);
        // Close file
        MPI_File_close(&inputFile);
    }

    // Initialize send/receive requests for old local block
    sendInit(old, grid, rowType, colType, send_a_request);
    recvInit(old, grid, rowType, colType, recv_a_request);

    // Initialize send/receive requests for current local block
    sendInit(current, grid, rowType, colType, send_b_request);
    recvInit(current, grid, rowType, colType, recv_b_request);

    // (συγχρονισμός διεργασιών πριν τις μετρήσεις)
    MPI_Barrier(grid.gridComm);

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();
    MPI_Pcontrol(1);

#pragma omp parallel num_threads(2) \
private(thread_id, temp, i, ss, stepLocalThreadChanges, buffer) \
shared(ompi_mpi_comm_world, ompi_mpi_info_null, ompi_mpi_comm_self, ompi_mpi_char, ompi_mpi_int, ompi_mpi_op_sum, \
threads, old, current, grid, stepLocalChanges, stepGlobalChanges, recv_a_status, recv_b_status, recv_a_request, \
send_a_request, recv_b_request, send_b_request, send_a_status, send_b_status, outputFolder, subArrayType, outputFile, \
fileRequests, fileStatus) reduction(+:s) \
default(none)
    {
        thread_id = omp_get_thread_num();
        threads = omp_get_num_threads();
        printf("rank: %d, worker: %d, threads: %d\n", grid.gridRank, thread_id, threads);

        // Start loop ...
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (ss = 1; ss <= STEPS; ss++) {
            s += 1;

            // Start receive/send requests
#pragma omp master
            {
                if (ss % 2) {
                    MPI_Startall(8, recv_a_request);
                    MPI_Startall(8, send_a_request);
                } else {
                    MPI_Startall(8, recv_b_request);
                    MPI_Startall(8, send_b_request);
                }
            };

            // Calculate internals
#pragma omp for schedule(static, 1) collapse(2) reduction(+:stepLocalChanges)
            for (i = 2; i < grid.localBlockDims[0]; i++) {
                for (j = 2; j < grid.localBlockDims[1]; j++) {
                    calculate(old, current, i, j, &stepLocalThreadChanges);
                    stepLocalChanges += stepLocalThreadChanges;
                }
            }

            // Wait receive requests
#pragma omp master
            {
                if (ss % 2) {
                    MPI_Waitall(8, recv_a_request, recv_a_status);
                } else {
                    MPI_Waitall(8, recv_b_request, recv_b_status);
                }
            }

            // Calculate up row
#pragma omp for schedule(static, 1) reduction(+:stepLocalChanges)
            for (i = 1; i < grid.localBlockDims[1] + 1; i++) {
                calculate(old, current, 1, i, &stepLocalThreadChanges);
                stepLocalChanges += stepLocalThreadChanges;
            }

            // Calculate down row
#pragma omp for schedule(static, 1) reduction(+:stepLocalChanges)
            for (i = 1; i < grid.localBlockDims[1] + 1; i++) {
                calculate(old, current, grid.localBlockDims[0], i, &stepLocalThreadChanges);
                stepLocalChanges += stepLocalThreadChanges;
            }

            // Calculate left Column
#pragma omp for schedule(static, 1) reduction(+:stepLocalChanges)
            for (i = 2; i < grid.localBlockDims[0]; i++) {
                calculate(old, current, i, 1, &stepLocalThreadChanges);
                stepLocalChanges += stepLocalThreadChanges;
            }


            // Calculate right column
#pragma omp for schedule(static, 1) reduction(+:stepLocalChanges)
            for (i = 2; i < grid.localBlockDims[0]; i++) {
                calculate(old, current, i, grid.localBlockDims[1], &stepLocalThreadChanges);
                stepLocalChanges += stepLocalThreadChanges;
            }

/*#pragma omp barrier
#pragma omp master
            print_step(ss, &grid, old, current);*/

#pragma omp barrier

#pragma omp master
            {
                // Create & write generation file
                sprintf(buffer, "%sstep-%d.txt", outputFolder, ss);
                MPI_File_open(MPI_COMM_SELF, buffer, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);
                MPI_File_set_view(outputFile, 0, MPI_CHAR, subArrayType, "native", MPI_INFO_NULL);
                for (i = 1; i <= grid.localBlockDims[0]; i++) {
                    MPI_File_iwrite(outputFile, &current[i][1], grid.localBlockDims[1], MPI_CHAR, &fileRequests[i - 1]);
                }

                // Wait until writing is done
                MPI_Waitall(grid.localBlockDims[0], fileRequests, fileStatus);

                // Close generation file
                MPI_File_close(&outputFile);
            }

#pragma omp single
            {
                // Swap local blocks
                temp = old;
                old = current;
                current = temp;
            }

#pragma omp master
            // Summarize all local changes
            if (ss % 10 == 0) {
                MPI_Allreduce(&stepLocalChanges, &stepGlobalChanges, 1, MPI_INT, MPI_SUM, grid.gridComm);
                if (stepGlobalChanges == 0) {
                    //break;
                }
            }

            // Wait send requests
#pragma omp master
            {
                if (ss % 2) {
                    MPI_Waitall(8, send_a_request, send_a_status);
                } else {
                    MPI_Waitall(8, send_b_request, send_b_status);
                }
            }
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    }

    // End MPI_Wtime
    end_w_time = MPI_Wtime();

    MPI_Pcontrol(0);

    local_time = end_w_time - start_w_time;

    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, grid.gridComm);

    printf("Worker: %d - Duration: %.9f\n", grid.gridRank, end_w_time - start_w_time);

    if (grid.gridRank == root) {
        printf("Steps: %d, Max time: %f\n", (s - 1) / threads, max_time);
        if (inputFileNotExists) {
            free2DArray(block, grid.blockDims[0]);
        }
//        system("/home/msi/projects/CLionProjects/game-of-life/mpi+openmp/scripts/clion-boxes.sh");
    }

    free2DArray(old, grid.localBlockDims[0] + 2);
    free2DArray(current, grid.localBlockDims[0] + 2);
    free(fileRequests);
    free(fileStatus);

    MPI_Comm_free(&grid.gridComm);

    MPI_Finalize();
}
