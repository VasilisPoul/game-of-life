#ifndef __MPI_H__
#define __MPI_H__

#include <mpi.h>
#include <stdbool.h>

#define UP_LEFT 0
#define UP 1
#define UP_RIGHT 2
#define RIGHT 3
#define DOWN_RIGHT 4
#define DOWN 5
#define DOWN_LEFT 6
#define LEFT 7

typedef struct neighbors {
    int up_left;
    int up;
    int up_right;
    int right;
    int down_right;
    int down;
    int down_left;
    int left;
} Neighbors;

typedef struct gridInfo {
    MPI_Comm gridComm;      // communicator for entire grid
    Neighbors neighbors;    // neighbor processes
    int gridRank;           // rank of current process in gridComm
    int dims[2];            // grid dimensions
    int row;                // row of current process
    int col;                // column of current process
    int processes;          // total number of processes
    int M;                  // N
    int N;                  // M
    int dimM;               // local M dimension
    int dimN;               // local N dimension
} GridInfo;

void printGridInfo(GridInfo *grid);

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
void initNeighbors(GridInfo *grid);

int setupGrid(GridInfo *grid, int N, int M);

int scatter2DArray(bool **array, bool **local, int root, GridInfo *grid);

int gather2DArray(bool **array, bool **local, int root, GridInfo *grid);

#endif
