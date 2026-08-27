#include "myCuda.h"
#include <stdio.h>
#include <stdlib.h>
#define N 8

int main(){
    MyCudaItem item;
    MyCudaMalloc(&item,N);
    item.src_buffer[0] = 1;
    item.src_buffer[1] = 2;
    item.src_buffer[2] = 3;
    item.src_buffer[3] = 4;
    item.src_buffer[4] = 2;
    item.src_buffer[5] = 3;
    item.src_buffer[6] = 1;
    item.src_buffer[7] = 4;
    MyCudaPrefixSum(&item);
}