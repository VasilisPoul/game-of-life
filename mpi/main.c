#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include "mpi.h"
#include "game_of_life.h"

#define STEPS 1000

int main(int argc, char **argv) {
    int s = 0, i = 0, j = 0, rank, size = 0, workers = 0, root = 0, sum = 0;
    double start_w_time = 0.0, end_w_time = 0.0, local_time = 0.0, max_time = 0.0;
    bool **block = NULL, **a = NULL, **b = NULL, **c = NULL;
    MPI_Datatype colType, rowType;
    MPI_Request send_a_request[8], recv_a_request[8], send_b_request[8], recv_b_request[8];
    MPI_Status send_a_status[8], send_b_status[8], recv_a_status[8], recv_b_status[8];

    GridInfo grid;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //TODO: Setup grid dimensions & local array dimensions based on blockDims[0], M, W parameters.
    setupGrid(&grid, TABLE_N, TABLE_M);

    if (rank == root) {
        //printGridInfo(&grid);
        block = allocate2DArray(grid.blockDims[0], grid.blockDims[1]);
        initialize_array(block, grid.blockDims[0], grid.blockDims[1]);
        printf("block:\n");
        print_array(block, true, true, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
                    grid.localBlockDims[1]);
    }

    a = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);
    b = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);

    for (i = 0; i < grid.localBlockDims[0] + 2; i++) {
        for (j = 0; j < grid.localBlockDims[1] + 2; j++) {
            a[i][j] = 0;
            b[i][j] = 0;
        }
    }

    // rowType
    MPI_Type_vector(1, grid.localBlockDims[1], 0, MPI_CXX_BOOL, &rowType);
    MPI_Type_commit(&rowType);

    // colType
    MPI_Type_vector(grid.localBlockDims[0], 1, grid.localBlockDims[1] + 2, MPI_CXX_BOOL, &colType);
    MPI_Type_commit(&colType);


    // Todo: read from file
    scatter2DArray(block, a, root, &grid);

    sendInit(a, grid, rowType, colType, send_a_request);
    recvInit(a, grid, rowType, colType, recv_a_request);

    sendInit(b, grid, rowType, colType, send_b_request);
    recvInit(b, grid, rowType, colType, recv_b_request);

    MPI_Barrier(grid.gridComm); //(συγχρονισμός διεργασιών πριν τις μετρήσεις)

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Start loop
    for (s = 0; s < STEPS; s++) {

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
            for (j = 2; j < grid.localBlockDims[1]; j++) {
                b[i][j] = 0;
                calculate(a, b, i, j, &grid.stepLocalChanges);
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
            b[1][j] = 0;
            calculate(a, b, 1, j, &grid.stepLocalChanges);
        }

        // Calculate down row
        for (j = 1; j < grid.localBlockDims[1] + 1; j++) {
            b[grid.localBlockDims[0]][j] = 0;
            calculate(a, b, grid.localBlockDims[0], j, &grid.stepLocalChanges);
        }

        // Calculate left Column
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            b[i][1] = 0;
            calculate(a, b, i, 1, &grid.stepLocalChanges);
        }

        // Calculate right column
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            b[i][grid.localBlockDims[1]] = 0;
            calculate(a, b, i, grid.localBlockDims[1], &grid.stepLocalChanges);
        }

        //print_step(s, &grid, a, b);
        // Todo: write to file
        gather2DArray(block, b, root, &grid);
//        for (i = 0; i < grid.processes; i++) {
//            MPI_Barrier(grid.gridComm);
//            if (i == grid.gridRank) {
//                if (grid.gridRank == root) {
//                    print_array(block, true, false, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
//                                grid.localBlockDims[1]);
//                }
//            }
//        }

        // Swap local blocks
        c = a;
        a = b;
        b = c;

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

    printf("Worker %d ==> Start Time = %.6f End Time = %.6f Duration = %.9f seconds\n", grid.gridRank, start_w_time, end_w_time, end_w_time - start_w_time);

    gather2DArray(block, a, root, &grid);

    //MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();

    if (rank == root) {
        printf("block:\n");
        print_array(block, true, true, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0],
                    grid.localBlockDims[1]);
        printf("Steps: %d, Max time: %f\n", s, max_time);
        free2DArray(block, grid.blockDims[0]);
    }

    free2DArray(a, grid.localBlockDims[0] + 2);
    free2DArray(b, grid.localBlockDims[0] + 2);
}
