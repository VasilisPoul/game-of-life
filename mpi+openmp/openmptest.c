#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>

int main(){
    
    
    int phase, n = 10, i;
    #pragma omp parallel default(none) shared(n) private(i, phase)
    {

        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();
       
        for (phase = 0; phase < n; phase++){
            
            printf("OUTER %d for tid:%d, total: %d \n\n", phase, tid, total);


            #pragma omp for
            for (i = 1; i < n; i+=2){
                printf("IN %d for tid:%d, total: %d\n",i , tid, total);
            }
        }
    }
    return 0;
}