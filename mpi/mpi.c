#include "mpi.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>


void printGridInfo(GridInfo *grid) {
    printf("\n");
    printf("Grid rank: %d\n", grid->gridRank);
    printf("Grid dimensions: (%d,%d)\n", grid->dims[0], grid->dims[1]);
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

int setupGrid(GridInfo *grid, int M, int N) {
    int flag = 0, worldRank = 0, periods[2] = {true, true}, gridCoords[2];

    // if MPI has not been initialized, abort procedure
    MPI_Initialized(&flag);
    if (flag == false)
        return -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->processes));

    grid->dims[0] = grid->dims[1] = (int) sqrt(grid->processes);

    //MPI_Dims_create(grid->processes, 2, grid->dims);

    // create communicator for the process grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid->dims, periods, true, &(grid->gridComm));

    // retrieve the process rank in the grid Communicator
    // and the process coordinates in the cartesian topology
    MPI_Comm_rank(grid->gridComm, &(grid->gridRank));

    MPI_Cart_coords(grid->gridComm, grid->gridRank, 2, gridCoords);
    grid->row = gridCoords[0];
    grid->col = gridCoords[1];

    grid->M = M;
    grid->N = N;

    grid->localM = M / (int) sqrt(grid->processes) + 2;
    grid->localN = N / (int) sqrt(grid->processes) + 2;

    // Initialize neighbors
    initNeighbors(grid);
    return 0;
}
