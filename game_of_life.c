#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int** create(int n, int m){
    
  int ** array = malloc(n * sizeof(int*));
  for (int i = 0; i < n; i++){
    array[i] = malloc(m * sizeof(int));
  }

  return array;
}

void initialize(int** array, int n, int m){
  srand(time(NULL));
  
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++){
      array[i][j] = rand() % 2;
    }
  }
}

void print_array(int** array, int n, int m){
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++){
      printf("%s ", array[i][j] ? "\u2B1B" : "\u2B1C");
    }
    
    printf("\n");
  }
}