#include "mpi.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

/**
 * Print grid info
*/
void printGridInfo(GridInfo *grid) {
    printf("\n------------------------------\n");
    printf("Grid rank: %d\n", grid->gridRank);
    printf("Grid dimensions: (%d,%d)\n", grid->dims[0], grid->dims[1]);
    printf("Number of Processes is %d\n", grid->processes);
    printf("Grid Comm Identifier is %d\n", grid->gridComm);
    printf("Current Process Coordinates are (%d, %d)\n", grid->row, grid->col);
    printf("Array dimensions: (%d, %d)\n", grid->N, grid->M);
    printf("Local array dimensions: (%d, %d)\n", grid->dimN, grid->dimM);
    printf("up neighbor: %d, down neighbor: %d\n", grid->neighbors.up, grid->neighbors.down);
    printf("left neighbor: %d, right neighbor: %d\n", grid->neighbors.left, grid->neighbors.right);
    printf("up left neighbor: %d, up right neighbor: %d\n", grid->neighbors.up_left, grid->neighbors.up_right);
    printf("down left neighbor: %d, down right neighbor: %d\n", grid->neighbors.down_left, grid->neighbors.down_right);
    printf("------------------------------\n");
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
    int flag = 0, worldRank = 0, periods[2] = {true, true}, gridCoords[2];

    // if MPI has not been initialized, abort procedure
    MPI_Initialized(&flag);
    if (flag == false)
        return -1;

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->processes));

    // Cartesian dimensions
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

    grid->N = N;
    grid->M = M;

    // Local array dimensions
    grid->dimM = M / (int) sqrt(grid->processes);
    grid->dimN = N / (int) sqrt(grid->processes);

    // Initialize neighbors
    initNeighbors(grid);
    return 0;
}

/**
 * Scatter 2D array
*/
int scatter2DArray(bool **array, bool **local, int root, GridInfo *grid) {
    int flag, rank, loops = grid->N / grid->dimN;
    int size, dest, packPosition;
    int c, index, coords[2], i, j;
    bool *tempArray;
    MPI_Status status;

    MPI_Initialized(&flag);
    if (flag == false) return (-1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((root < 0) || (root >= size)) return (-1);

    tempArray = (bool *) malloc(grid->dimN * grid->dimN * sizeof(bool));
    if (!tempArray) return (-1);

    if (rank == root) {
        for (c = 0; c < loops; c++) {
            coords[0] = c;
            for (index = 0; index < loops; index++) {
                coords[1] = index;
                MPI_Cart_rank(grid->gridComm, coords, &dest);
                packPosition = 0;
                for (i = grid->dimN * c; i < grid->dimN * (c + 1); i++) {
                    for (j = grid->dimN * index; j < grid->dimN * (index + 1); j++) {
                        MPI_Pack(&array[i][j], 1, MPI_CXX_BOOL, tempArray, 256, &packPosition, grid->gridComm);
                    }
                }
                if (dest != root) {
                    MPI_Send(tempArray, grid->dimN * grid->dimN, MPI_CXX_BOOL, dest, 0, grid->gridComm);
                } else {
                    for (i = 1; i < grid->dimN + 1; i++)
                        for (j = 1; j < grid->dimM + 1; j++)
                            local[i][j] = tempArray[(i - 1) * grid->dimN + (j - 1)];
                }
            }
        }
    } else {
        MPI_Recv(tempArray, grid->dimN * grid->dimN, MPI_CXX_BOOL, root, 0, grid->gridComm, &status);
        for (c = 1; c < grid->dimN + 1; c++)
            for (index = 1; index < grid->dimM + 1; index++)
                local[c][index] = tempArray[(c - 1) * grid->dimN + (index - 1)];
    }
    free(tempArray);
    return (0);
}

/**
 * Gather 2D array
*/
int gather2DArray(bool **array, bool **local, int root, GridInfo *grid) {
    int flag, rank, loops = grid->N / grid->dimN;
    int size, cnt, source, rootCoords[2];
    int counter, index, coords[2], i, j;
    bool *tempArray;
    MPI_Status status;
    MPI_Initialized(&flag);
    if (flag == false) return (-1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((root < 0) || (root >= size)) return (-1);

    tempArray = (bool *) malloc(grid->dimN * grid->dimN * sizeof(bool));
    if (!tempArray) return (-1);
    if (rank == root) {
        for (counter = 0; counter < loops; counter++) {
            coords[0] = counter;
            for (index = 0; index < loops; index++) {
                coords[1] = index;
                MPI_Cart_rank(grid->gridComm, coords, &source);
                if (source == root) {
                    MPI_Cart_coords(grid->gridComm, rank, 2, rootCoords);
                    for (i = grid->dimN * rootCoords[0]; i < grid->dimN * (rootCoords[0] + 1); i++) {
                        for (j = grid->dimN * rootCoords[1]; j < grid->dimN * (rootCoords[1] + 1); j++) {
                            array[i][j] = local[i+1][j+1];
                        }
                    }
                } else {
                    MPI_Recv(tempArray, grid->dimN * grid->dimN, MPI_CXX_BOOL, source, 0, grid->gridComm, &status);
                    cnt = 0;
                    for (i = grid->dimN * counter; i < grid->dimN * (counter + 1); i++) {
                        for (j = grid->dimN * index; j < grid->dimN * (index + 1); j++) {
                            array[i][j] = tempArray[cnt];
                            cnt++;
                        }
                    }
                }
            }
        }
    } else {
        cnt = 0;
        for (i = 1; i < grid->dimN + 1; i++)
            for (j = 1; j < grid->dimM + 1; j++) {
                tempArray[cnt] = local[i][j];
                cnt++;
            }
        MPI_Send(tempArray, grid->dimN * grid->dimN, MPI_CXX_BOOL, root, 0, grid->gridComm);
    }
    free(tempArray);
    return (0);
}
