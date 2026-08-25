#include "myCuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>

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
    for(int i = 0; i<cuda_item->buffer_size;i++){
        cuda_item->dest_buffer[i] = 0; 
    }
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

    assert(cuda_item && "CudaItem non valido");

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



int MyCudaIterations(MyCudaItem* item){
    /*calcola le iterazioni sul buffer per completarlo, arrotondando all'intero superiore*/
    assert(item && "CudaItem non valido");
    return (item->buffer_size+BLOCK_DIM-1)/BLOCK_DIM;
}

void* MyCudaReductionPass(void* args){
    ThreadArgs* info = (ThreadArgs*) args;

    int idx = info->threadIdx+info->blockIdx*BLOCK_DIM;
    int size = info->size;
    int* src = info->src;
    int* dest = info->dest;

    //printf("THREAD %d\n",idx);

    if(idx >= size){
        pthread_exit(NULL);
    }
    printf("dest[%d] = %d+%d\n",idx,src[2*idx],src[2*idx+1]);
    dest[idx] = src[2*idx]+src[2*idx+1];
    pthread_exit(NULL);
}

void MyCudaReduction(MyCudaItem* cuda_item){
    assert(cuda_item && "Cuda_item non valido");
    int passes = MyCudaIterations(cuda_item);
    printf("PASSI: %d\n",passes);
    int res_size = cuda_item->buffer_size;
    res_size = (res_size+1)/2;
    pthread_t threads[BLOCK_DIM];
    ThreadArgs thread_arguments[BLOCK_DIM];
    while(1){
        printf("DESTSIZE: %d\n",res_size);
        for(int bid = 0; bid<passes;bid++){
            //printf("BLOCK %d\n",bid);
            for(int tid = 0; tid < BLOCK_DIM; tid++){
                //printf("CREO THREAD %d\n",tid);
                thread_arguments[tid].blockIdx = bid;
                thread_arguments[tid].dest = cuda_item->dest_buffer;
                thread_arguments[tid].src = cuda_item->src_buffer;
                thread_arguments[tid].size = res_size;
                thread_arguments[tid].threadIdx = tid;

                //kernel launch:
                pthread_create(&threads[tid],NULL,MyCudaReductionPass,&thread_arguments[tid]);
            }
            for(int tid = 0; tid < BLOCK_DIM; tid++){
                pthread_join(threads[tid],NULL);
            }
            
        }
        if(res_size>1){
            res_size = (res_size+1)/2;
            MyCudaMemSwap(cuda_item);
        }else{
            break;
        }
    }
    printf("RES=%d\n",cuda_item->dest_buffer[0]);
}
