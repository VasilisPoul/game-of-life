#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "mpi.h"
#include "game_of_life.h"


#define STEPS 1

int main(int argc, char **argv) {
    int s = 0, i = 0, j = 0, rank, size = 0, workers = 0, root = 0, sum = 0, rc = 0, array_of_sizes[2], array_of_subsizes[2], starts[2];
    double start_w_time = 0.0, end_w_time = 0.0, local_time = 0.0, max_time = 0.0;
    char **block = NULL, **old = NULL, **current = NULL, **temp = NULL;
    MPI_Datatype colType, rowType, subType;
    MPI_Request send_a_request[8], recv_a_request[8], send_b_request[8], recv_b_request[8];
    MPI_Status send_a_status[8], send_b_status[8], recv_a_status[8], recv_b_status[8];
    MPI_Status status;
    MPI_Request *readFileReq;
    MPI_File fh;
    GridInfo grid;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //TODO: Setup grid dimensions & local array dimensions based on blockDims[0], M, W parameters.
    setupGrid(&grid, TABLE_N, TABLE_M);

    // Allocate local blocks
    old = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);
    current = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);

    // Initialize local blocks
    for (i = 0; i < grid.localBlockDims[0] + 2; i++) {
        for (j = 0; j < grid.localBlockDims[1] + 2; j++) {
            old[i][j] = '0';
            current[i][j] = '0';
        }
    }

    // rowType
    MPI_Type_vector(1, grid.localBlockDims[1], 0, MPI_CHAR, &rowType);
    MPI_Type_commit(&rowType);

    // colType
    MPI_Type_vector(grid.localBlockDims[0], 1, grid.localBlockDims[1] + 2, MPI_CHAR, &colType);
    MPI_Type_commit(&colType);

    // Open input file
    rc = MPI_File_open(MPI_COMM_WORLD, "/home/msi/projects/CLionProjects/game-of-life/mpi/input/12x12.txt",
                       MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    if (rc) {
        // No file, generate array
        if (rank == root) {
            //printGridInfo(&grid);
            block = allocate2DArray(grid.blockDims[0], grid.blockDims[1]);
            initialize_block(block, grid.blockDims[0], grid.blockDims[1]);
            printf("block:\n");
            print_array(block, true, true, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
                        grid.localBlockDims[1]);
        }
        scatter2DArray(block, old, root, &grid);
    } else {
        // Read from file

        array_of_sizes[0] = grid.blockDims[0] + 1;
        array_of_sizes[1] = grid.blockDims[1] + 1;

        array_of_subsizes[0] = grid.localBlockDims[0];
        array_of_subsizes[1] = grid.localBlockDims[1];

        starts[0] = grid.gridCoords[0] * grid.localBlockDims[0];
        starts[1] = grid.gridCoords[1] * grid.localBlockDims[1];

        MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, starts, MPI_ORDER_C, MPI_CHAR, &subType);
        MPI_Type_commit(&subType);

        MPI_File_set_view(fh, 0, MPI_CHAR, subType, "native", MPI_INFO_NULL);

        readFileReq = malloc(grid.localBlockDims[0] * sizeof(MPI_Request));

        for (i = 1; i < 4; i++) {
            MPI_File_iread(fh, &old[i][1], 3, MPI_BYTE, &readFileReq[i - 1]);
        }

        // Wait until reading is done
        MPI_Waitall(3, readFileReq, &status);
        print_step(s, &grid, old, current);

        MPI_File_close(&fh);
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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Start loop
    for (s = 0; s < STEPS; s++) {

//        char file_name[100];
//        sprintf(file_name, "./generations/%dGeneration.txt", i + 1);
//
//        /*Create a file with the output of each iteration*/
//        if (grid.gridRank == 0) {
//            MPI_File gen;
//            MPI_File_open(MPI_COMM_SELF, file_name, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &gen);
//            MPI_File_close(&gen);
//        }


        // Start receive/send requests
        if (s % 2 == 0) {
            MPI_Startall(8, recv_a_request);
            MPI_Startall(8, send_a_request);
        } else {
            MPI_Startall(8, recv_b_request);
            MPI_Startall(8, send_b_request);
        }

        grid.stepLocalChanges = 0;

        // Calculate internals
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            for (j = 2; j < grid.localBlockDims[
                    1]; j++) {
                calculate(old, current, i, j, &grid.stepLocalChanges);
            }
        }

        // Wait receive requests
        if (s % 2 == 0) {
            MPI_Waitall(8, recv_a_request, recv_a_status);
        } else {
            MPI_Waitall(8, recv_b_request, recv_b_status);
        }

        // Calculate up row
        for (j = 1; j < grid.localBlockDims[1] + 1; j++) {
            calculate(old, current, 1, j, &grid.stepLocalChanges);
        }

        // Calculate down row
        for (j = 1; j < grid.localBlockDims[1] + 1; j++) {
            calculate(old, current, grid.localBlockDims[0], j, &grid.stepLocalChanges);
        }

        // Calculate left Column
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            calculate(old, current, i, 1, &grid.stepLocalChanges);
        }

        // Calculate right column
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            calculate(old, current, i, grid.localBlockDims[1], &grid.stepLocalChanges);
        }











        if(rc){
            //print_step(s, &grid, old, current);
            // Todo: write to file
            gather2DArray(block, current, root, &grid);
            for (i = 0; i < grid.processes; i++) {
                MPI_Barrier(grid.gridComm);
                if (i == grid.gridRank) {
                    if (grid.gridRank == root) {
                        print_array(block, true, false, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
                                    grid.localBlockDims[1]);
                    }
                }
            }
        }


        /*Write the next generation array to a file*/
//        MPI_File ge;
//        MPI_Status ge_status;
//        //MPI_File_open(new_comm, file_name, MPI_MODE_WRONLY, MPI_INFO_NULL, &ge);
//        MPI_File_set_view(ge, 0, MPI_CHAR, mysubarray, "native", MPI_INFO_NULL);
//        for (j = 1; j < grid.localBlockDims[0] + 1; j++)
//            MPI_File_write_all(ge, &current[j][1], grid.localBlockDims[0], MPI_CHAR, &ge_status);
//        MPI_File_close(&ge);



        // Swap local blocks
        temp = old;
        old = current;
        current = temp;

        // Summarize local all local changes
        if (s % 10 == 0) {
            MPI_Allreduce(&grid.stepLocalChanges, &grid.stepGlobalChanges, 1, MPI_INT, MPI_SUM, grid.gridComm);
            if (grid.stepGlobalChanges == 0) {
                break;
            }
        }

        // Wait send requests
        if (s % 2 == 0) {
            MPI_Waitall(8, send_a_request, send_a_status);
        } else {
            MPI_Waitall(8, send_b_request, send_b_status);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //End MPI_Wtime
    end_w_time = MPI_Wtime();
    local_time = end_w_time - start_w_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, grid.gridComm);

    printf("Worker %d ==> Start Time = %.6f End Time = %.6f Duration = %.9f seconds\n", grid.gridRank, start_w_time,
           end_w_time, end_w_time - start_w_time);


    if (rc && rank == root) {

        gather2DArray(block, old, root, &grid);

        printf("block:\n");
        print_array(block, true, true, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
                    grid.localBlockDims[1]);
        printf("Steps: %d, Max time: %f\n", s, max_time);
        free2DArray(block, grid.blockDims[0]);
    }

    free2DArray(old, grid.localBlockDims[0] + 2);
    free2DArray(current, grid.localBlockDims[0] + 2);

    //MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();
}
