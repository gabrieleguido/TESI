#include "myCuda.h"
#include <stdio.h>
#include <stdlib.h>
#define N 16

int main(int argc, char* argv[]){
    MyCudaItem item;
    int n = N;
    if(argc > 1){
        n = atoi(argv[1]);
    }
    MyCudaMalloc(&item,n);

    MyCudaPrefixSumTest(&item,RANDOM_RANGE);
}