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
    MPI_Type_vector(grid.dimN, 1, grid.dimM, MPI_DOUBLE, &colType);
    MPI_Type_commit(&colType);

    scatter2DArray(state, localState, root, &grid);

    printf("localState:\n");
    print_array(localState, grid.dimN, grid.dimM, grid.dimN, grid.dimM);


    MPI_Barrier(grid.gridComm); //(συγχρονισμός διεργασιών πριν τις μετρήσεις)

    // Start MPI_Wtime
    start_w_time = MPI_Wtime();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for (s = 0; s < STEPS; s++) { //For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
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
