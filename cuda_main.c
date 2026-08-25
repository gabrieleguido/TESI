#include "myCuda.h"
#include <stdio.h>
#define N 101

int main(){
    MyCudaItem item;
    MyCudaMalloc(&item,N);
    for(int i = 0;i<item.buffer_size;i++){
        item.src_buffer[i] = i+1;
        item.dest_buffer[i] = 0;
    }
    MyCudaBuffersPrint(&item);
    MyCudaReduction(&item);
    MyCudaBuffersPrint(&item);
}