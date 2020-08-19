#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include "mpi.h"
#include "game_of_life.h"

#define STEPS 2

int main(int argc, char **argv) {
    int s = 0, i = 0, j = 0, rank, size = 0, workers = 0, root = 0, alive = 0, stepAlive = 0, stepChanges = 0, stepLocalChanges = 0, sum = 0;
    double start_w_time = 0.0, end_w_time = 0.0,  local_time = 0.0,  max_time = 0.0;
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
        print_array(block, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0], grid.localBlockDims[1]);
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

    scatter2DArray(block, a, root, &grid);

    sendInit(a, grid, rowType, colType, send_a_request);
    recvInit(a, grid, rowType, colType, recv_a_request);

    sendInit(b, grid, rowType, colType, send_b_request);
    recvInit(b, grid, rowType, colType, recv_b_request);

    MPI_Barrier(grid.gridComm); //(συγχρονισμός διεργασιών πριν τις μετρήσεις)

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
    for (s = 0; s < STEPS; s++) {

        // Irecv(RRequest) //X 8(Β, Ν, Δ, Α + γωνιακά)
        // Isend (SRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)
        if (s % 2 == 0) {
            MPI_Startall(8, recv_a_request);
            MPI_Startall(8, send_a_request);
        } else {
            MPI_Startall(8, recv_b_request);
            MPI_Startall(8, send_b_request);
        }

        stepLocalChanges = 0;

        // Υπολογισμός εσωτερικών στοιχείων «μετά» (άσπρα στοιχεία στο σχήμα) και ένδειξη κενού πίνακα ή μη αλλαγής (διπλό for)
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            for (j = 2; j < grid.localBlockDims[1]; j++) {
                calculate(a, b, i, j, &stepLocalChanges);
            }
        }

        // Wait (RRequest) Χ 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of RRequests)
        if (s % 2 == 0) {
            MPI_Waitall(8, recv_a_request, recv_a_status);
        } else {
            MPI_Waitall(8, recv_b_request, recv_b_status);
        }



        // Υπολογισμός εξωτερικών στοιχείων «μετά» (πράσινα στο σχήμα)-4 for
        // Up row
        for (j = 1; j < grid.localBlockDims[1] + 1; j++) {
            calculate(a, b, 1, j, &stepLocalChanges);
        }

        // Down row
        for (j = 1; j < grid.localBlockDims[1] + 1; j++) {
            calculate(a, b, grid.localBlockDims[0], j, &stepLocalChanges);
        }

        // left Column
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            calculate(a, b, i, 1, &stepLocalChanges);
        }

        // right column
        for (i = 2; i < grid.localBlockDims[0]; i++) {
            calculate(a, b, i, grid.localBlockDims[1], &stepLocalChanges);
        }

        for (i = 0; i < size; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                printf("step: %d\n", s);
                printGridInfo(&grid);
                printf("a:");
                print_array(a, grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2, grid.localBlockDims[0] + 2,
                            grid.localBlockDims[1] + 2);
                printf("alive: %d, changes: %d\n", stepAlive, stepLocalChanges);

                printf("b:");
                print_array(b, grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2,
                            grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);
            }
        }

        // Πριν Πίνακας = Μετά Πίνακας SWAP
        c = a;
        a = b;
        b = c;

        // (reduce για έλεγχο εδώ)
        if (s % 10 == 0) {
            MPI_Allreduce(&stepLocalChanges, &stepChanges, 1, MPI_INT, MPI_SUM, grid.gridComm);
            if (stepChanges == 0) {
                printf("step: %d, rank: %d, stepLocalChanges: %d, stepChanges: %d\n", s,
                       grid.gridRank, stepLocalChanges, stepChanges);
                break;
            }
        }

        // Wait(SRequest) X 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of SRequests)
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

    //printf("Worker %d ==> Start Time = %.6f End Time = %.6f Duration = %.9f seconds\n", grid.gridRank, start_w_time, end_w_time, end_w_time - start_w_time);

    if (s % 2 == 0) {
        gather2DArray(block, a, root, &grid);
    } else {
        gather2DArray(block, b, root, &grid);
    }

    //MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();

    if (rank == root) {
        printf("block:\n");
        print_array(block, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0], grid.localBlockDims[1]);
        printf("Max time: %f\n", max_time);
        free2DArray(block, grid.blockDims[0]);
    }

    free2DArray(a, grid.localBlockDims[0] + 2);
    free2DArray(b, grid.localBlockDims[0] + 2);
}
