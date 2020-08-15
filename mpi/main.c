#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include "mpi.h"
#include "game_of_life.h"

#define STEPS 20

int main(int argc, char **argv) {
    int s = 0, i = 0, j = 0, rank, size = 0, workers = 0, root = 0;
    double start_w_time = 0.0, end_w_time = 0.0;
    bool **state = NULL, **localOldState = NULL, **localState = NULL;
    MPI_Datatype colType, rowType;
    MPI_Request request[16];
    MPI_Status status[16];
    GridInfo grid;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //TODO: Setup grid dimensions & local array dimensions based on N, M, W parameters.
    setupGrid(&grid, TABLE_N, TABLE_M);

    printf("World rank: %d, grid rank: %d, size: %d\n", rank, grid.gridRank, size);

    if (rank == root) {
        printGridInfo(&grid);
        state = allocate2DArray(grid.N, grid.M);
        initialize_array(state, grid.N, grid.M);
        printf("state:\n");
        print_array(state, grid.N, grid.M, grid.dimN, grid.dimM);
    }

    localOldState = allocate2DArray(grid.dimN + 2, grid.dimM + 2);
    localState = allocate2DArray(grid.dimN + 2, grid.dimM + 2);

    for (i = 0; i < grid.dimN + 2; i++) {
        for (j = 0; j < grid.dimM + 2; j++) {
            localOldState[i][j] = 0;
            localState[i][j] = 0;
        }
    }

    //rowType
    MPI_Type_vector(1, grid.dimM, 0, MPI_CXX_BOOL, &rowType);
    MPI_Type_commit(&rowType);

    //colType
    MPI_Type_vector(grid.dimN, 1, grid.dimM, MPI_CXX_BOOL, &colType);
    MPI_Type_commit(&colType);

    scatter2DArray(state, localState, root, &grid);

    printf("localState:\n");
    print_array(localState, grid.dimN, grid.dimM, grid.dimN, grid.dimM);


    MPI_Barrier(grid.gridComm); //(συγχρονισμός διεργασιών πριν τις μετρήσεις)

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();

    MPI_Send_init(&localState[1][1], 1, rowType, grid.neighbors.up, UP, grid.gridComm, &request[0]);
    MPI_Send_init(&localState[grid.dimN - 2][1], 1, rowType, grid.neighbors.down, DOWN, grid.gridComm, &request[1]);
    MPI_Send_init(&localState[1][grid.dimM - 2], 1, colType, grid.neighbors.right, RIGHT, grid.gridComm, &request[2]);
    MPI_Send_init(&localState[1][1], 1, colType, grid.neighbors.left, LEFT,  grid.gridComm, &request[3]);

    MPI_Send_init(&localState[1][grid.dimM - 2], 1, MPI_CXX_BOOL,  grid.neighbors.up_right, UP_RIGHT, grid.gridComm, &request[4]);
    MPI_Send_init(&localState[1][1], 1, MPI_CXX_BOOL, grid.neighbors.up_left, UP_LEFT,  grid.gridComm, &request[5]);
    MPI_Send_init(&localState[grid.dimN - 2][grid.dimM - 2], 1, MPI_CXX_BOOL, grid.neighbors.down_right, DOWN_RIGHT, grid.gridComm, &request[6]);
    MPI_Send_init(&localState[grid.dimN - 2][1], 1, MPI_CXX_BOOL, grid.neighbors.down_left, DOWN_LEFT, grid.gridComm, &request[7]);

    MPI_Recv_init(&localState[grid.dimN - 1][1], 1, rowType, grid.neighbors.down, DOWN, grid.gridComm, &request[8]);
    MPI_Recv_init(&localState[0][1], 1, rowType, grid.neighbors.up, UP, grid.gridComm, &request[9]);
    MPI_Recv_init(&localState[1][0], 1, colType, grid.neighbors.left, LEFT, grid.gridComm, &request[10]);
    MPI_Recv_init(&localState[1][grid.dimM - 1], 1, colType, grid.neighbors.right, RIGHT, grid.gridComm, &request[11]);

    MPI_Recv_init(&localState[grid.dimN - 1][0], 1, MPI_CXX_BOOL, grid.neighbors.down_left, DOWN_LEFT, grid.gridComm, &request[12]);
    MPI_Recv_init(&localState[grid.dimN - 1][grid.dimM - 1], 1, MPI_CXX_BOOL, grid.neighbors.down_right, DOWN_RIGHT, grid.gridComm, &request[13]);
    MPI_Recv_init(&localState[0][0], 1, MPI_CXX_BOOL, grid.neighbors.up_left, UP_LEFT, grid.gridComm, &request[14]);
    MPI_Recv_init(&localState[0][grid.dimM - 1], 1, MPI_CXX_BOOL, grid.neighbors.up_right, UP_RIGHT, grid.gridComm, &request[15]);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (s = 0; s < STEPS; s++) { //For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
        MPI_Startall(16, request);
        //Irecv(RRequest) //X 8(Β, Ν, Δ, Α + γωνιακά)

//        Isend (SRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)
//        Υπολογισμός εσωτερικών στοιχείων «μετά» (άσπρα στοιχεία στο σχήμα) και
//        ένδειξη κενού πίνακα ή μη αλλαγής (διπλό for)
//        Wait (RRequest) Χ 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of RRequests)
//        Υπολογισμός εξωτερικών στοιχείων «μετά» (πράσινα στο σχήμα)-4 for
//        Πριν Πίνακας = Μετά Πίνακας
//        (reduce για έλεγχο εδώ)
//        Wait(SRequest) X 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of SRequests)
    } //End for

    //MPI_Reduce(&sum,&total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //End MPI_Wtime
    end_w_time = MPI_Wtime();

    printf("Worker %d ==> Send Time = %.6f Recv Time = %.6f Duration = %.9f seconds\n", grid.gridRank, start_w_time,
           end_w_time, end_w_time - start_w_time);


    gather2DArray(state, localState, root, &grid);


    if (rank == root) {
        printf("state:\n");
        print_array(state, grid.N, grid.M, grid.dimN, grid.dimM);
    }

    //MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();
}
