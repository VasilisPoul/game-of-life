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
    int processes;          // total number of processes
    int gridRank;           // rank of current process in gridComm
    int gridDims[2];        // grid dimensions
    int gridCoords[2];      // grid dimensions
    int blockDims[2];       // block dimensions
    int localBlockDims[2];  // local block dimensions
    int stepGlobalChanges;
    int stepLocalChanges;
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

void sendInit(bool **block, GridInfo grid, MPI_Datatype rowType, MPI_Datatype colType, MPI_Request *req);

void recvInit(bool **block, GridInfo grid, MPI_Datatype rowType, MPI_Datatype colType, MPI_Request *req);

int scatter2DArray(bool **array, bool **local, int root, GridInfo *grid);

int gather2DArray(bool **array, bool **local, int root, GridInfo *grid);

void print_step(int step, GridInfo *grid, bool **block_a, bool **block_b);

#endif
