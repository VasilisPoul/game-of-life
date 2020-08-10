#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "game_of_life.h"

#define STEPS 20

#define DIM 2

/**
 * Find nearest neighbors and save ranks on neighbors array:
    UP LEFT = 0
    UP = 1 !
    UP RIGHT = 2
    RIGHT = 3 !
    DOWN RIGHT = 4
    DOWN = 5 !
    DOWN LEFT = 6
    LEFT = 7 !
*/
void getNeighbors(MPI_Comm cartComm, int rank, int *neighbors) {
    int x = 0, y = 0, coords[DIM];
    MPI_Cart_coords(cartComm, rank, DIM, coords);
    x = coords[0];
    y = coords[1];

    /* Find up & down neighbor */
    MPI_Cart_shift(cartComm, 0, 1, &neighbors[1], &neighbors[5]);

    /* Find left & right neighbor */
    MPI_Cart_shift(cartComm, 1, 1, &neighbors[7], &neighbors[3]);

    /* Find up left neighbor */
    coords[0] = x - 1;
    coords[1] = y - 1;
    MPI_Cart_rank(cartComm, coords, &neighbors[0]);

    /* Find up right neighbor */
    coords[0] = x - 1;
    coords[1] = y + 1;
    MPI_Cart_rank(cartComm, coords, &neighbors[2]);

    /* Find down left neighbor */
    coords[0] = x + 1;
    coords[1] = y - 1;
    MPI_Cart_rank(cartComm, coords, &neighbors[6]);

    /* Find down right neighbor */
    coords[0] = x + 1;
    coords[1] = y + 1;
    MPI_Cart_rank(cartComm, coords, &neighbors[4]);
}

int main(int argc, char **argv) {
    MPI_Comm cartComm;
    int s, rank, cartRank, size, workers = 0, neighbors[8];
    double startWtime, endwtime;
    int array[N][M];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[DIM] = {0, 0};
    int periodic[DIM] = {true, true};
    MPI_Dims_create(size, DIM, dims);
    if (rank == 0)
        printf("dims={%d, %d}\n", dims[0], dims[1]);


    MPI_Cart_create(MPI_COMM_WORLD, DIM, dims, periodic, true, &cartComm);
    MPI_Comm_rank(cartComm, &cartRank);

    printf("rank = %d\n", rank);
    printf("cartRank = %d\n", cartRank);

    getNeighbors(cartComm, cartRank, neighbors);

    printf("UP: %d, DOWN: %d\n", neighbors[1], neighbors[5]);
    printf("LEFT: %d, RIGHT: %d\n", neighbors[7], neighbors[3]);
    printf("UP LEFT: %d, UP RIGHT: %d\n", neighbors[0], neighbors[2]);
    printf("DOWN LEFT: %d, DOWN RIGHT: %d\n", neighbors[6], neighbors[4]);
    printf("\n");

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

    MPI_Finalize();
}
