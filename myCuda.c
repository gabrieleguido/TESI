#include "myCuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void MyCudaMalloc(MyCudaItem* cuda_item, int n){
    /*alloca n blocchi int nei due buffer di cuda_item*/

    assert(cuda_item && "CudaItem non valido");
    int* res;
    res = malloc(n*sizeof(int));
    assert(res && "Errore allocazione memoria dest MyCuda");
    cuda_item->dest_buffer = res;
    res = malloc(n*sizeof(int));
    assert(res && "Errore allocazione memoria src MyCuda");
    cuda_item->src_buffer = res;
    cuda_item->buffer_size = n;
}
void MyCudaMemSwap(MyCudaItem* cuda_item){
    /*inverte dest e src di cuda item*/

    assert(cuda_item && "CudaItem non valido");
    int* aux = cuda_item->src_buffer;
    cuda_item->src_buffer = cuda_item->dest_buffer;
    cuda_item->dest_buffer = aux;
    printf("SWAP\n");
}


void MyCudaMemFree(MyCudaItem* cuda_item){
    /*cancella cuda item e i suoi buffer*/

    assert(cuda_item && "CudaItem non valido");
    free(cuda_item->dest_buffer);
    free(cuda_item->src_buffer);
    free(cuda_item);
}

void MyCudaBuffersPrint(MyCudaItem* cuda_item){
    int size = cuda_item->buffer_size;
    int* src = cuda_item->src_buffer;
    int* dest = cuda_item->dest_buffer;
    printf("----------------------\n");
    printf("SRC:\n");
    for(int i = 0; i<size;i++){
        printf("src[%d] = %d,\n",i,src[i]);
    }
    printf("DEST:\n");
    for(int i = 0; i<size;i++){
        printf("dest[%d] = %d,\n",i,dest[i]);
    }
    printf("----------------------\n");
}
