#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "game_of_life.h"

#define STEPS 20

typedef struct Neighbors {
    int up_left;
    int up;
    int up_right;
    int right;
    int down_right;
    int down;
    int down_left;
    int left;
} Neighbors;

typedef struct GridInfo {
    MPI_Comm gridComm;      // communicator for entire grid
    Neighbors neighbors;    // neighbor processes
    int gridRank;       // rank of current process in gridComm
    int row;                // row of current process
    int col;                // column of current process
    int processes;          // total number of processes
} GridInfo;

void printGridInfo(GridInfo *grid) {
    printf("\n");
    printf("Process rank in Grid is %d\n", grid->gridRank);
    printf("Number of Processes is %d\n", grid->processes);
    printf("Grid Comm Identifier is %d\n", grid->gridComm);
    printf("Current Process Coordinates are (%d, %d)\n", grid->row, grid->col);
    printf("up neighbor: %d, down neighbor: %d\n", grid->neighbors.up, grid->neighbors.down);
    printf("left neighbor: %d, right neighbor: %d\n", grid->neighbors.left, grid->neighbors.right);
    printf("up left neighbor: %d, up right neighbor: %d\n", grid->neighbors.up_left, grid->neighbors.up_right);
    printf("down left neighbor: %d, down right neighbor: %d\n", grid->neighbors.down_left, grid->neighbors.down_right);
    printf("\n");
}

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
void initNeighbors(GridInfo *grid) {
    int x = 0, y = 0, coords[2];
    MPI_Cart_coords(grid->gridComm, grid->gridRank, 2, coords);
    x = coords[0];
    y = coords[1];

    /* Find up & down neighbor */
    MPI_Cart_shift(grid->gridComm, 0, 1, &grid->neighbors.up, &grid->neighbors.down);

    /* Find left & right neighbor */
    MPI_Cart_shift(grid->gridComm, 1, 1, &grid->neighbors.left, &grid->neighbors.right);

    /* Find up left neighbor */
    coords[0] = x - 1;
    coords[1] = y - 1;
    MPI_Cart_rank(grid->gridComm, coords, &grid->neighbors.up_left);

    /* Find up right neighbor */
    coords[0] = x - 1;
    coords[1] = y + 1;
    MPI_Cart_rank(grid->gridComm, coords, &grid->neighbors.up_right);

    /* Find down left neighbor */
    coords[0] = x + 1;
    coords[1] = y - 1;
    MPI_Cart_rank(grid->gridComm, coords, &grid->neighbors.down_left);

    /* Find down right neighbor */
    coords[0] = x + 1;
    coords[1] = y + 1;
    MPI_Cart_rank(grid->gridComm, coords, &grid->neighbors.down_right);
}

int SetupGrid(GridInfo *grid) {
    int flag = 0, worldRank = 0, dims[2] = {0, 0}, periods[2] = {true, true}, gridCoords[2];

    // if MPI has not been initialized, abort procedure
    MPI_Initialized(&flag);
    if (flag == false)
        return -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->processes));

    MPI_Dims_create(grid->processes, 2, dims);

    // create communicator for the process grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &(grid->gridComm));

    // retrieve the process rank in the grid Communicator
    // and the process coordinates in the cartesian topology
    MPI_Comm_rank(grid->gridComm, &(grid->gridRank));

    MPI_Cart_coords(grid->gridComm, grid->gridRank, 2, gridCoords);
    grid->row = gridCoords[0];
    grid->col = gridCoords[1];

    // Initialize neighbors
    initNeighbors(grid);
    return 0;
}

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
