#include "myCuda.h"
#include <stdio.h>

int main(){
    MyCudaItem item;
    MyCudaMalloc(&item,10);
    for(int i = 0;i<item.buffer_size;i++){
        item.src_buffer[i] = 0;
        item.dest_buffer[i] = i;
    }
    MyCudaBuffersPrint(&item);
    MyCudaMemSwap(&item);
    MyCudaBuffersPrint(&item);
}