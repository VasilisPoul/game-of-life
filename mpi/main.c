#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "mpi.h"
#include <math.h>
#include "game_of_life.h"

#define STEPS 20

bool **Allocate2DMatrix(int rows, int columns) {
    int counter;
    bool **matrix;
    matrix = (bool **) malloc(rows * sizeof(bool *));
    if (!matrix)
        return (NULL);
    for (counter = 0; counter < rows; counter++) {
        matrix[counter] = (bool *) malloc(columns * sizeof(bool));
        if (!matrix[counter])
            return (NULL);
    }
    return matrix;
}

void Free2DMatrix(bool **matrix, int rows) {
    int counter;
    for (counter = 0; counter < rows; counter++)
        free((bool *) matrix[counter]);
    free((bool **) matrix);
}

int main(int argc, char **argv) {
    int rank = 0, size = 0, workers = 0;
    double start_w_time = 0.0, end_w_time = 0.0;

    GridInfo grid;

    double **localA, **localB, **localC;
    double **aMatrix, **bMatrix, **cMatrix;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SetupGrid(&grid);

    printf("World rank: %d, grid rank: %d, size: %d\n", rank, grid.gridRank, size);

    if (rank == 0) {
        printGridInfo(&grid);
    }


    MPI_Comm_free(&grid.gridComm);
    MPI_Finalize();


    /*
     * workers = sqrt((double) size);
     *
     * MPI_Barrier (συγχρονισμός διεργασιών πριν τις μετρήσεις)
        Start MPI_Wtime
        For #επαναλήψεων (σταθερός σε όλες τις μετρήσεις σας)
        Irecv (RRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)
        Isend (SRequest) X 8 (Β,Ν,Δ,Α + γωνιακά)
        Υπολογισμός εσωτερικών στοιχείων «μετά» (άσπρα στοιχεία στο σχήμα) και
        ένδειξη κενού πίνακα ή μη αλλαγής (διπλό for)
        Wait (RRequest) Χ 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of RRequests)
        Υπολογισμός εξωτερικών στοιχείων «μετά» (πράσινα στο σχήμα)-4 for
        Πριν Πίνακας = Μετά Πίνακας
        (reduce για έλεγχο εδώ)
        Wait(SRequest) X 8 (Β,Ν,Δ,Α+γωνιακά) ή WaitAll (array of SRequests)
        End for
        End MPI_Wtime
     */



    //todo Υπολογισμός γειτονικών διεργασιών με βοηθητική συνάρτηση.


//    MPI_Barrier(MPI_COMM_WORLD);
//
//    startWtime = MPI_Wtime();
//
//    for (s = 0; s < STEPS; s++) {
//
//
//    }
//
//    endwtime = MPI_Wtime();

//    if (size < 1) {
//        if (rank == 0)
//            printf("Invalid process number. Aborting...\n");
//        MPI_Abort(MPI_COMM_WORLD, -2);
//    }

    //print_array(array, N, M);
    //initialize(array, N, M);
    ///
    //operate(array, N, M);
    ///
}
