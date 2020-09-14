#include "mpi.h"
#include "game_of_life.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

/*COLOR*/
#define RED  "\x1B[31m"

/*BOLD-COLOR*/
#define B_RED  "\x1B[1m\x1B[31m"
#define B_GREEN  "\x1B[1m\x1B[32m"

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
    printf("Neighbors:\n");
    printf(B_GREEN" %2.2d   %2.2d    %2.2d \n"RESET, grid->neighbors.up_left, grid->neighbors.up,
           grid->neighbors.up_right);
    printf("   ↖   ↑   ↗\n");
    printf(B_GREEN"%2.2d"RESET" ←  "B_RED"%2.2d"RESET"   → "B_GREEN"%2.2d\n"RESET, grid->neighbors.left, grid->gridRank,
           grid->neighbors.right);
    printf("   ↙   ↓   ↘\n");
    printf(B_GREEN" %2.2d   %2.2d    %2.2d\n"RESET, grid->neighbors.down_left, grid->neighbors.down,
           grid->neighbors.down_right);
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
 * Send requests initialization
*/
void sendInit(char **block, GridInfo grid, MPI_Datatype rowType, MPI_Datatype colType, MPI_Request *req) {
    MPI_Send_init(&block[grid.localBlockDims[0]][1], 1, rowType, grid.neighbors.down, UP, grid.gridComm,
                  &req[0]);
    MPI_Send_init(&block[1][1], 1, rowType, grid.neighbors.up, DOWN, grid.gridComm, &req[1]);
    MPI_Send_init(&block[1][1], 1, colType, grid.neighbors.left, RIGHT, grid.gridComm, &req[2]);
    MPI_Send_init(&block[1][grid.localBlockDims[1]], 1, colType, grid.neighbors.right, LEFT, grid.gridComm,
                  &req[3]);
    MPI_Send_init(&block[grid.localBlockDims[0]][grid.localBlockDims[1]], 1, MPI_CHAR,
                  grid.neighbors.down_right, UP_LEFT, grid.gridComm, &req[4]);
    MPI_Send_init(&block[grid.localBlockDims[0]][1], 1, MPI_CHAR, grid.neighbors.down_left, UP_RIGHT,
                  grid.gridComm, &req[5]);
    MPI_Send_init(&block[1][1], 1, MPI_CHAR, grid.neighbors.up_left, DOWN_RIGHT, grid.gridComm, &req[6]);
    MPI_Send_init(&block[1][grid.localBlockDims[1]], 1, MPI_CHAR, grid.neighbors.up_right, DOWN_LEFT,
                  grid.gridComm, &req[7]);
}

/**
 * Recieve requests initialization
*/
void recvInit(char **block, GridInfo grid, MPI_Datatype rowType, MPI_Datatype colType, MPI_Request *req) {
    MPI_Recv_init(&block[0][1], 1, rowType, grid.neighbors.up, UP, grid.gridComm, &req[0]);
    MPI_Recv_init(&block[grid.localBlockDims[0] + 1][1], 1, rowType, grid.neighbors.down, DOWN, grid.gridComm,
                  &req[1]);
    MPI_Recv_init(&block[1][grid.localBlockDims[1] + 1], 1, colType, grid.neighbors.right, RIGHT, grid.gridComm,
                  &req[2]);
    MPI_Recv_init(&block[1][0], 1, colType, grid.neighbors.left, LEFT, grid.gridComm, &req[3]);
    MPI_Recv_init(&block[0][0], 1, MPI_CHAR, grid.neighbors.up_left, UP_LEFT, grid.gridComm, &req[4]);
    MPI_Recv_init(&block[0][grid.localBlockDims[1] + 1], 1, MPI_CHAR, grid.neighbors.up_right, UP_RIGHT,
                  grid.gridComm, &req[5]);
    MPI_Recv_init(&block[grid.localBlockDims[0] + 1][grid.localBlockDims[1] + 1], 1, MPI_CHAR,
                  grid.neighbors.down_right, DOWN_RIGHT, grid.gridComm, &req[6]);
    MPI_Recv_init(&block[grid.localBlockDims[0] + 1][0], 1, MPI_CHAR, grid.neighbors.down_left, DOWN_LEFT,
                  grid.gridComm, &req[7]);
}

/**
 * Scatter 2D array
*/
int scatter2DArray(char **array, char **local, int root, GridInfo *grid) {
    int flag, rank, loops = grid->blockDims[0] / grid->localBlockDims[0];
    int size, dest, packPosition;
    int c, index, coords[2], i, j;
    char *tempArray;
    MPI_Status status;

    MPI_Initialized(&flag);
    if (flag == false) return (-1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((root < 0) || (root >= size)) return (-1);

    tempArray = (char *) malloc(grid->localBlockDims[0] * grid->localBlockDims[0] * sizeof(char));
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
                        MPI_Pack(&array[i][j], 1, MPI_CHAR, tempArray, 256, &packPosition, grid->gridComm);
                    }
                }
                if (dest != root) {
                    MPI_Send(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CHAR, dest, 0,
                             grid->gridComm);
                } else {
                    for (i = 1; i < grid->localBlockDims[0] + 1; i++)
                        for (j = 1; j < grid->localBlockDims[1] + 1; j++)
                            local[i][j] = tempArray[(i - 1) * grid->localBlockDims[0] + (j - 1)];
                }
            }
        }
    } else {
        MPI_Recv(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CHAR, root, 0, grid->gridComm,
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
int gather2DArray(char **array, char **local, int root, GridInfo *grid) {
    int flag, rank, loops = grid->blockDims[0] / grid->localBlockDims[0];
    int size, cnt, source, rootCoords[2];
    int counter, index, coords[2], i, j;
    char *tempArray;
    MPI_Status status;
    MPI_Initialized(&flag);
    if (flag == false) return (-1);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((root < 0) || (root >= size)) return (-1);

    tempArray = (char *) malloc(grid->localBlockDims[0] * grid->localBlockDims[0] * sizeof(char));
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
                    MPI_Recv(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CHAR, source, 0,
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
        MPI_Send(tempArray, grid->localBlockDims[0] * grid->localBlockDims[0], MPI_CHAR, root, 0, grid->gridComm);
    }
    free(tempArray);
    return (0);
}

void print_step(int step, GridInfo *grid, char **old, char **current) {
    int i;
    for (i = 0; i < grid->processes; i++) {
        MPI_Barrier(grid->gridComm);
        if (i == grid->gridRank) {
            printf("----------------------------------------------------------------------------------------\n");
            printf("step: %d\n", step);
            printGridInfo(grid);
            printf("old:");
            print_array(old, false, true, grid->localBlockDims[0] + 2, grid->localBlockDims[1] + 2,
                        grid->localBlockDims[0] + 2,
                        grid->localBlockDims[1] + 2);
            printf("current:");
            print_array(current, false, true, grid->localBlockDims[0] + 2, grid->localBlockDims[1] + 2,
                        grid->localBlockDims[0] + 2,
                        grid->localBlockDims[1] + 2);
        }
    }
}
