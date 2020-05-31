#include <stdio.h>

#include "game_of_life.h"

#define N 10;
#define M 10;

int main(int argc, char** argv){
  int n = N;
  int m = M;
  
  int** array = create(n, m);

  print_array(array, n, m);
  printf("\n");
  initialize(array, n, m);

  /*print initialized array*/
  
  print_array(array, n, m);

  operate( array, n, m);
}