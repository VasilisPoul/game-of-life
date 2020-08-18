#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include "mpi.h"
#include "game_of_life.h"

#define STEPS 1

int main(int argc, char **argv) {
    int s = 0, i = 0, j = 0, rank, size = 0, workers = 0, root = 0;
    double start_w_time = 0.0, end_w_time = 0.0;
    bool **block = NULL, **localOldBlock = NULL, **localBlock = NULL;
    MPI_Datatype colType, rowType;
    GridInfo grid;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //TODO: Setup grid dimensions & local array dimensions based on blockDims[0], M, W parameters.
    setupGrid(&grid, TABLE_N, TABLE_M);

    printf("World rank: %d, grid rank: %d, size: %d\n", rank, grid.gridRank, size);

    if (rank == root) {
        printGridInfo(&grid);
        block = allocate2DArray(grid.blockDims[0], grid.blockDims[1]);
        initialize_array(block, grid.blockDims[0], grid.blockDims[1]);
        printf("block:\n");
        print_array(block, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0], grid.localBlockDims[1]);
    }

    localOldBlock = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);
    localBlock = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);

    for (i = 0; i < grid.localBlockDims[0] + 2; i++) {
        for (j = 0; j < grid.localBlockDims[1] + 2; j++) {
            localOldBlock[i][j] = 0;
            localBlock[i][j] = 0;
        }
    }

    //rowType
    MPI_Type_vector(1, grid.localBlockDims[1], 0, MPI_CXX_BOOL, &rowType);
    MPI_Type_commit(&rowType);

    //colType
    MPI_Type_vector(grid.localBlockDims[0], 1, grid.localBlockDims[1] + 2, MPI_CXX_BOOL, &colType);
    MPI_Type_commit(&colType);

    scatter2DArray(block, localBlock, root, &grid);

    //printf("localBlock:\n");
    //print_array(localBlock, grid.localBlockDims[0], grid.localBlockDims[1], grid.localBlockDims[0], grid.localBlockDims[1]);

    MPI_Request sendRequest[8], recvRequest[8];
    MPI_Status status[8];

    MPI_Send_init(&localBlock[grid.localBlockDims[0]][1], 1, rowType, grid.neighbors.down, UP, grid.gridComm,
                  &sendRequest[0]);
    MPI_Recv_init(&localBlock[0][1], 1, rowType, grid.neighbors.up, UP, grid.gridComm, &recvRequest[0]);

    MPI_Send_init(&localBlock[1][1], 1, rowType, grid.neighbors.up, DOWN, grid.gridComm, &sendRequest[1]);
    MPI_Recv_init(&localBlock[grid.localBlockDims[0] + 1][1], 1, rowType, grid.neighbors.down, DOWN, grid.gridComm,
                  &recvRequest[1]);

    // Todo: fix
    //MPI_Send_init(&localBlock[1][1], 1, colType, grid.neighbors.left, RIGHT, grid.gridComm, &sendRequest[0]);
    //MPI_Recv_init(&localBlock[1][grid.localBlockDims[1] + 1], 1, colType, grid.neighbors.right, RIGHT, grid.gridComm, &recvRequest[0]);

    // Todo: fix
    //MPI_Send_init(&localBlock[1][grid.localBlockDims[1]], 1, colType, grid.neighbors.right, LEFT, grid.gridComm, &sendRequest[0]);
    //MPI_Recv_init(&localBlock[1][0], 1, colType, grid.neighbors.left, LEFT, grid.gridComm, &recvRequest[0]);

    MPI_Send_init(&localBlock[grid.localBlockDims[0]][grid.localBlockDims[1]], 1, MPI_CXX_BOOL,
                  grid.neighbors.down_right, UP_LEFT,
                  grid.gridComm,
                  &sendRequest[2]);
    MPI_Recv_init(&localBlock[0][0], 1, MPI_CXX_BOOL, grid.neighbors.up_left, UP_LEFT, grid.gridComm, &recvRequest[2]);

    MPI_Send_init(&localBlock[grid.localBlockDims[0]][1], 1, MPI_CXX_BOOL, grid.neighbors.down_left, UP_RIGHT,
                  grid.gridComm,
                  &sendRequest[3]);
    MPI_Recv_init(&localBlock[0][grid.localBlockDims[1] + 1], 1, MPI_CXX_BOOL, grid.neighbors.up_right, UP_RIGHT,
                  grid.gridComm,
                  &recvRequest[3]);

    MPI_Send_init(&localBlock[1][1], 1, MPI_CXX_BOOL, grid.neighbors.up_left, DOWN_RIGHT, grid.gridComm,
                  &sendRequest[4]);
    MPI_Recv_init(&localBlock[grid.localBlockDims[0] + 1][grid.localBlockDims[1] + 1], 1, MPI_CXX_BOOL,
                  grid.neighbors.down_right,
                  DOWN_RIGHT,
                  grid.gridComm, &recvRequest[4]);

    MPI_Send_init(&localBlock[1][grid.localBlockDims[1]], 1, MPI_CXX_BOOL, grid.neighbors.up_right, DOWN_LEFT,
                  grid.gridComm,
                  &sendRequest[5]);
    MPI_Recv_init(&localBlock[grid.localBlockDims[0] + 1][0], 1, MPI_CXX_BOOL, grid.neighbors.down_left, DOWN_LEFT,
                  grid.gridComm,
                  &recvRequest[5]);

    MPI_Barrier(grid.gridComm); //(συγχρονισμός διεργασιών πριν τις μετρήσεις)

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (s = 0; s < STEPS; s++) { //For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
        MPI_Startall(1, recvRequest); // Irecv(RRequest) //X 8(Β, Ν, Δ, Α + γωνιακά)
        MPI_Startall(1, sendRequest);  // Isend (SRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)

        // Υπολογισμός εσωτερικών στοιχείων «μετά» (άσπρα στοιχεία στο σχήμα) και
        // ένδειξη κενού πίνακα ή μη αλλαγής (διπλό for)
        // Wait (RRequest) Χ 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of RRequests)

        MPI_Waitall(1, recvRequest, status);
        MPI_Waitall(1, sendRequest, status);

        for (i = 0; i < size; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                printGridInfo(&grid);
                printf("localBlock array:");
                print_array(localBlock, grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2,
                            grid.localBlockDims[0] + 2,
                            grid.localBlockDims[1] + 2);
            }
        }

        // Υπολογισμός εξωτερικών στοιχείων «μετά» (πράσινα στο σχήμα)-4 for
        // Πριν Πίνακας = Μετά Πίνακας
        // (reduce για έλεγχο εδώ)
        // Wait(SRequest) X 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of SRequests)
    }

    // MPI_Reduce(&sum,&total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //End MPI_Wtime
    end_w_time = MPI_Wtime();

    printf("Worker %d ==> Start Time = %.6f End Time = %.6f Duration = %.9f seconds\n", grid.gridRank, start_w_time,
           end_w_time, end_w_time - start_w_time);

    gather2DArray(block, localBlock, root, &grid);

    if (rank == root) {
        printf("block:\n");
        print_array(block, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0], grid.localBlockDims[1]);
    }

    //MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();
}
