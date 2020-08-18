#include "mpi.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

/*COLOR*/
#define B_U_GREEN  "\x1B[1m\x1B[32m"

/*RESET COLOR*/
#define RESET  "\x1B[0m"

/**
 * Print grid info
*/
void printGridInfo(GridInfo *grid) {
    printf("Number of Processes: %d\n", grid->processes);
    printf("Grid dimensions: (%d,%d)\n", grid->gridDims[0], grid->gridDims[1]);
    printf("Process rank on grid: %d\n", grid->gridRank);
    printf("Process coordinates: (%d, %d)\n", grid->gridCoords[0], grid->gridCoords[1]);
    printf("Block dimensions: (%d, %d)\n", grid->blockDims[0], grid->blockDims[1]);
    printf("Local block dimensions: (%d, %d)\n", grid->localBlockDims[0], grid->localBlockDims[1]);
    printf(" %2.2d   %2.2d    %2.2d \n", grid->neighbors.up_left, grid->neighbors.up, grid->neighbors.up_right);
    printf("   ↖   ↑   ↗\n");
    printf("%2.2d ←  "B_U_GREEN"%2.2d"RESET"   → %2.2d\n", grid->neighbors.left, grid->gridRank, grid->neighbors.right);
    printf("   ↙   ↓   ↘\n");
    printf(" %2.2d   %2.2d    %2.2d\n", grid->neighbors.down_left, grid->neighbors.down, grid->neighbors.down_right);
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

/**
 * Setup grid object
*/
int setupGrid(GridInfo *grid, int N, int M) {
    int flag = 0, worldRank = 0, periods[2] = {true, true};

    // if MPI has not been initialized, abort procedure
    MPI_Initialized(&flag);
    if (flag == false)
        return -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->processes));

    // Cartesian dimensions
    grid->gridDims[0] = grid->gridDims[1] = (int) sqrt(grid->processes);

    //MPI_Dims_create(grid->processes, 2, grid->gridDims);

    // create communicator for the process grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid->gridDims, periods, true, &(grid->gridComm));

    // retrieve the process rank in the grid Communicator
    // and the process coordinates in the cartesian topology
    MPI_Comm_rank(grid->gridComm, &(grid->gridRank));

    MPI_Cart_coords(grid->gridComm, grid->gridRank, 2, grid->gridCoords);

    grid->blockDims[0] = N;
    grid->blockDims[1] = M;

    // Local array dimensions
    grid->localBlockDims[0] = N / (int) sqrt(grid->processes);
    grid->localBlockDims[1] = M / (int) sqrt(grid->processes);

    // Initialize neighbors
    initNeighbors(grid);
    return 0;
}

/**
 * Scatter 2D array
*/
int scatter2DArray(bool **array, bool **local, int root, GridInfo *grid) {
    int flag, rank, loops = grid->blockDims[0] / grid->localBlockDims[0];
    int size, dest, packPosition;
    int c, index, coords[2], i, j;
    bool *tempArray;
    MPI_Status status;

    MPI_Initialized(&flag);
    if (flag == false) return (-1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((root < 0) || (root >= size)) return (-1);

    tempArray = (bool *) malloc(grid->localBlockDims[0] * grid->localBlockDims[0] * sizeof(bool));
    if (!tempArray) return (-1);

    if (rank == root) {
        for (c = 0; c < loops; c++) {
            coords[0] = c;
            for (index = 0; index < loops; index++) {
                coords[1] = index;
                MPI_Cart_rank(grid->gridComm, coords, &dest);
                packPosition = 0;
                for (i = grid->localBlockDims[0] * c; i < grid->localBlockDims[0] * (c + 1); i++) {
                    for (j = grid->localBlockDims[0] * index; j < grid->localBlockDims[0] * (index + 1); j++) {
                        MPI_Pack(&array[i][j], 1, MPI_CXX_BOOL, tempArray, 256, &packPosition, grid->gridComm);
                    }
                }
                if (dest != root) {
                    MPI_Send(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CXX_BOOL, dest, 0,
                             grid->gridComm);
                } else {
                    for (i = 1; i < grid->localBlockDims[0] + 1; i++)
                        for (j = 1; j < grid->localBlockDims[1] + 1; j++)
                            local[i][j] = tempArray[(i - 1) * grid->localBlockDims[0] + (j - 1)];
                }
            }
        }
    } else {
        MPI_Recv(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CXX_BOOL, root, 0, grid->gridComm,
                 &status);
        for (c = 1; c < grid->localBlockDims[0] + 1; c++)
            for (index = 1; index < grid->localBlockDims[1] + 1; index++)
                local[c][index] = tempArray[(c - 1) * grid->localBlockDims[0] + (index - 1)];
    }
    free(tempArray);
    return (0);
}

/**
 * Gather 2D array
*/
int gather2DArray(bool **array, bool **local, int root, GridInfo *grid) {
    int flag, rank, loops = grid->blockDims[0] / grid->localBlockDims[0];
    int size, cnt, source, rootCoords[2];
    int counter, index, coords[2], i, j;
    bool *tempArray;
    MPI_Status status;
    MPI_Initialized(&flag);
    if (flag == false) return (-1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((root < 0) || (root >= size)) return (-1);

    tempArray = (bool *) malloc(grid->localBlockDims[0] * grid->localBlockDims[0] * sizeof(bool));
    if (!tempArray) return (-1);
    if (rank == root) {
        for (counter = 0; counter < loops; counter++) {
            coords[0] = counter;
            for (index = 0; index < loops; index++) {
                coords[1] = index;
                MPI_Cart_rank(grid->gridComm, coords, &source);
                if (source == root) {
                    MPI_Cart_coords(grid->gridComm, rank, 2, rootCoords);
                    for (i = grid->localBlockDims[0] * rootCoords[0];
                         i < grid->localBlockDims[0] * (rootCoords[0] + 1); i++) {
                        for (j = grid->localBlockDims[0] * rootCoords[1];
                             j < grid->localBlockDims[0] * (rootCoords[1] + 1); j++) {
                            array[i][j] = local[i + 1][j + 1];
                        }
                    }
                } else {
                    MPI_Recv(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CXX_BOOL, source, 0,
                             grid->gridComm,
                             &status);
                    cnt = 0;
                    for (i = grid->localBlockDims[0] * counter; i < grid->localBlockDims[0] * (counter + 1); i++) {
                        for (j = grid->localBlockDims[0] * index; j < grid->localBlockDims[0] * (index + 1); j++) {
                            array[i][j] = tempArray[cnt];
                            cnt++;
                        }
                    }
                }
            }
        }
    } else {
        cnt = 0;
        for (i = 1; i < grid->localBlockDims[0] + 1; i++)
            for (j = 1; j < grid->localBlockDims[1] + 1; j++) {
                tempArray[cnt] = local[i][j];
                cnt++;
            }
        MPI_Send(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CXX_BOOL, root, 0, grid->gridComm);
    }
    free(tempArray);
    return (0);
}
