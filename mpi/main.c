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
    MPI_Request sendRequest[8], sendOldRequest[8], recvRequest[8], recvOldRequest[8];
    MPI_Status sendStatus[8], sendOldStatus[8], recvStatus[8], recvOldStatus[8];

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

    localOldBlock = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);
    localBlock = allocate2DArray(grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);

    for (i = 0; i < grid.localBlockDims[0] + 2; i++) {
        for (j = 0; j < grid.localBlockDims[1] + 2; j++) {
            localOldBlock[i][j] = 0;
            localBlock[i][j] = 0;
        }
    }

    // rowType
    MPI_Type_vector(1, grid.localBlockDims[1], 0, MPI_CXX_BOOL, &rowType);
    MPI_Type_commit(&rowType);

    // colType
    MPI_Type_vector(grid.localBlockDims[0], 1, grid.localBlockDims[1] + 2, MPI_CXX_BOOL, &colType);
    MPI_Type_commit(&colType);

    // Scatter block in localBlocks
    scatter2DArray(block, localBlock, root, &grid);

    scatter2DArray(block, localOldBlock, root, &grid);

    sendInit(localBlock, grid, rowType, colType, sendRequest);
    sendInit(localOldBlock, grid, rowType, colType, recvOldRequest);

    recvInit(localBlock, grid, rowType, colType, recvRequest);
    recvInit(localOldBlock, grid, rowType, colType, sendOldRequest);

    MPI_Barrier(grid.gridComm); //(συγχρονισμός διεργασιών πριν τις μετρήσεις)

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (s = 0; s < STEPS; s++) { //For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
        MPI_Startall(8, recvRequest); // Irecv(RRequest) //X 8(Β, Ν, Δ, Α + γωνιακά)
        MPI_Startall(8, sendRequest);  // Isend (SRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)

        // Todo: Υπολογισμός εσωτερικών στοιχείων «μετά» (άσπρα στοιχεία στο σχήμα) και
        //  ένδειξη κενού πίνακα ή μη αλλαγής (διπλό for)

        MPI_Waitall(8, recvRequest, recvStatus); // Wait (RRequest) Χ 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of RRequests)

        // Todo: Υπολογισμός εξωτερικών στοιχείων «μετά» (πράσινα στο σχήμα)-4 for

        // Todo: Πριν Πίνακας = Μετά Πίνακας

        for (i = 0; i < size; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {
                printGridInfo(&grid);
                printf("localBlock:");
                print_array(localBlock, grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2,
                            grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2);

                printf("localOldBlock:");
                print_array(localOldBlock, grid.localBlockDims[0] + 2, grid.localBlockDims[1] + 2,
                            grid.localBlockDims[0] + 2,
                            grid.localBlockDims[1] + 2);
            }
        }

        // (reduce για έλεγχο εδώ)

        MPI_Waitall(8, sendRequest, sendStatus); // Wait(SRequest) X 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of SRequests)
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //End MPI_Wtime
    end_w_time = MPI_Wtime();

    //printf("Worker %d ==> Start Time = %.6f End Time = %.6f Duration = %.9f seconds\n", grid.gridRank, start_w_time, end_w_time, end_w_time - start_w_time);

    gather2DArray(block, localBlock, root, &grid);

    if (rank == root) {
        printf("block:\n");
        print_array(block, grid.blockDims[0], grid.blockDims[1], grid.localBlockDims[0], grid.localBlockDims[1]);
    }

    //MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();
}
