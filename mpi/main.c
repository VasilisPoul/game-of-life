#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include <time.h>
#include "game_of_life.h"

#define STEPS 20

int main(int argc, char **argv) {
    GridInfo grid;
    int i = 0, j = 0, rank = 0, size = 0, workers = 0;
    double start_w_time = 0.0, end_w_time = 0.0;
    bool **state = NULL, **localPrevious = NULL, **localNext = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //TODO: Setup grid dimensions & local array dimensions based on N, M, W parameters.
    setupGrid(&grid, TABLE_N, TABLE_M);

    printf("World rank: %d, grid rank: %d, size: %d\n", rank, grid.gridRank, size);

    if (rank == 0) {
        printGridInfo(&grid);
        state = allocate2DArray(grid.N, grid.M);
        initialize_array(state, grid.N, grid.M);
        print_array(state, grid.N, grid.M);
    }

    localPrevious = allocate2DArray(grid.localN, grid.localM);
    localNext = allocate2DArray(grid.localN, grid.localM);

    for (i = 0; i < grid.localN; i++) {
        for (j = 0; j < grid.localM; j++) {
            localPrevious[i][j] = 0;
            localNext[i][j] = 0;
        }
    }

    print_array(localPrevious, grid.localN, grid.localM);


    //MPI_Scatter()


//    * MPI_Barrier (συγχρονισμός διεργασιών πριν τις μετρήσεις)
//    Start MPI_Wtime
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
        for (s = 0; s < STEPS; s++) { //For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
            Irecv (RRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)
            Isend (SRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)
            Υπολογισμός εσωτερικών στοιχείων «μετά» (άσπρα στοιχεία στο σχήμα) και
            ένδειξη κενού πίνακα ή μη αλλαγής (διπλό for)
            Wait (RRequest) Χ 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of RRequests)
            Υπολογισμός εξωτερικών στοιχείων «μετά» (πράσινα στο σχήμα)-4 for
            Πριν Πίνακας = Μετά Πίνακας
            (reduce για έλεγχο εδώ)
            Wait(SRequest) X 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of SRequests)
        } //End for
     */

    //MPI_Reduce(&sum,&total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //End MPI_Wtime


    MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();
}
