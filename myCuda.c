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
    /*inverte dest e src di cuda item, AZZERANDO dest*/

    assert(cuda_item && "CudaItem non valido");
    int* aux = cuda_item->src_buffer;
    cuda_item->src_buffer = cuda_item->dest_buffer;
    cuda_item->dest_buffer = aux;
    for(int i = 0; i<cuda_item->buffer_size;i++){
        cuda_item->dest_buffer[i] = 0; 
    }
    // printf("SWAP\n");
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
void BuffersPrint(int* src, int* dest, int size){

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

void* MyCudaReductionKernel(void* args){
    ThreadArgs* info = (ThreadArgs*) args;

    int idx = info->threadIdx+info->blockIdx*BLOCK_DIM;
    int size = info->size;
    int* src = info->src;
    int* dest = info->dest;

    //printf("THREAD %d\n",idx);

    if(idx >= size){
        pthread_exit(NULL);
    }
    // printf("dest[%d] = %d+%d\n",idx,src[2*idx],src[2*idx+1]);
    dest[idx] = src[2*idx]+src[2*idx+1];
    pthread_exit(NULL);
}

void MyCudaReduction(MyCudaItem* cuda_item){
    assert(cuda_item && "Cuda_item non valido");
 
    cuda_item->size = cuda_item->buffer_size;
    cuda_item->size = (cuda_item->size+1)/2;
    cuda_item->kernel_fn = MyCudaReductionKernel;

    while(1){
        MyCudaKernelLaunch(cuda_item);
        if(cuda_item->size>1){
            cuda_item->size = (cuda_item->size+1)/2;
            MyCudaMemSwap(cuda_item);
        }else{
            break;
        }
    }
    printf("RES=%d\n",cuda_item->dest_buffer[0]);
}

void MyCudaSetThreadArgs(ThreadArgs* targs,int threadIdx, int blockIdx, int* dest, int* src, int size){
    /*prepara gli argomenti per il kernel launch*/
    assert(targs && "ThreadArgs non valida");
    targs->threadIdx = threadIdx;
    targs->blockIdx = blockIdx;
    targs->dest = dest;
    targs->src = src;
    targs->size = size;
}

void MyCudaKernelLaunch(MyCudaItem* cuda_item){
    assert(cuda_item && "Cuda_item non valido");

    int passes = MyCudaIterations(cuda_item);
    // printf("PASSI: %d\n",passes);
    pthread_t threads[BLOCK_DIM];
    ThreadArgs thread_arguments[BLOCK_DIM];
    for(int bid = 0; bid<passes;bid++){
            //printf("BLOCK %d\n",bid);
            for(int tid = 0; tid < BLOCK_DIM; tid++){
                //printf("CREO THREAD %d\n",tid);
                MyCudaSetThreadArgs(&thread_arguments[tid],tid,bid,cuda_item->dest_buffer,cuda_item->src_buffer,cuda_item->size);

                //kernel launch:
                pthread_create(&threads[tid],NULL,cuda_item->kernel_fn,&thread_arguments[tid]);
            }
            for(int tid = 0; tid < BLOCK_DIM; tid++){
                pthread_join(threads[tid],NULL);
            }
            
        }
}

int MyCudaIsPowerTwo(int n){
    int res = n;
    while(res>2){
        if(res%2){
            return 0;
        }
        res = res/2;
    }
    return 1;
}

void MyCudaPrefixSum(MyCudaItem* cuda_item){
    assert(cuda_item && "Cuda_item non valido");
    assert(MyCudaIsPowerTwo(cuda_item->buffer_size) && "Size deve essere potenza di 2");
 
    cuda_item->size = cuda_item->buffer_size;
    cuda_item->size = (cuda_item->size+1)/2;
    cuda_item->kernel_fn = MyCudaPrefixDownPassKernel;

    if(DEBUG)MyCudaBuffersPrint(cuda_item);

    int* orig_dets = cuda_item->dest_buffer;
    int* orig_src = cuda_item->src_buffer;

    if(DEBUG)printf("DOWNPASS------\n");
    while(cuda_item->size>0){
        if(DEBUG)printf("SIZE = %d\n",cuda_item->size);
        MyCudaKernelLaunch(cuda_item);
        if(DEBUG)BuffersPrint(orig_src,orig_dets,cuda_item->buffer_size);
        if(cuda_item->size == 1){
            break;
        }
        cuda_item->src_buffer = cuda_item->dest_buffer;
        cuda_item->dest_buffer += cuda_item->size; 
        cuda_item->size = (cuda_item->size+1)/2;
    }
    if(DEBUG) printf("dest'[0] = %d,src'[0] = %d\n",cuda_item->dest_buffer[0],cuda_item->src_buffer[0]);

    //size = 1 perchè siamo appena usciti dal while
    cuda_item->size = 2;

    cuda_item->dest_buffer = cuda_item->src_buffer;

    cuda_item->kernel_fn = MyCudaPrefixUpPassKernel;


    if(DEBUG)printf("UPPASS------\n");

    while(cuda_item->size*2 < cuda_item->buffer_size){
        if(DEBUG)printf("size = %d",cuda_item->size);
        cuda_item->src_buffer = cuda_item->dest_buffer;
        cuda_item->dest_buffer -= cuda_item->size*2;
        if(DEBUG)printf("dest'[0] = %d,src'[0] = %d\n",cuda_item->dest_buffer[0],cuda_item->src_buffer[0]);
        MyCudaKernelLaunch(cuda_item);
        if(DEBUG)BuffersPrint(orig_src,orig_dets,cuda_item->buffer_size);
        cuda_item->size *= 2;
    }

    cuda_item->dest_buffer = orig_src;
    cuda_item->src_buffer = orig_dets;

    //ultima chiamata per quando siamo tornati su 2 buffer diversi
    MyCudaKernelLaunch(cuda_item);

    if(DEBUG)MyCudaBuffersPrint(cuda_item);
}
void* MyCudaPrefixDownPassKernel(void* args){
    ThreadArgs* info = (ThreadArgs*) args;

    int idx = info->threadIdx+info->blockIdx*BLOCK_DIM;
    int size = info->size;
    int* src = info->src;
    int* dest = info->dest;
    
    //printf("THREAD %d\n",idx);
    
    if(idx >= size){
        pthread_exit(NULL);
    }

    dest[idx] = src[2*idx]+src[2*idx +1];
    if(DEBUG)printf(">>>dest[%d] = %d+%d\n",idx,src[2*idx],src[2*idx+1]);
    src[2*idx +1] = dest[idx];
    if(DEBUG)printf(">>>src[%d] = %d\n",2*idx+1,src[2*idx +1]);


}

void* MyCudaPrefixUpPassKernel(void* args){
    ThreadArgs* info = (ThreadArgs*) args;

    int idx = info->threadIdx+info->blockIdx*BLOCK_DIM;
    int size = info->size;
    int* src = info->src;
    int* dest = info->dest;
    
    //printf("THREAD %d\n",idx);
    
    if(idx >= size){
        pthread_exit(NULL);
    }

    if(2*(idx+1) < size*2){
        if(DEBUG)printf(">>>dest[%d] = %d+%d\n",2*(idx+1),dest[2*(idx+1)],src[idx]);
        dest[2*(idx+1)] += src[idx];
    }
    if(2*(idx+1)+1 < size*2){
        if(DEBUG)printf(">>>dest[%d] = %d+%d\n",2*(idx+1)+1,dest[2*(idx+1)+1],src[idx]);
        dest[2*(idx+1)+1] += src[idx];
    }


}

void MyCudaPrefixSumTest(MyCudaItem* cuda_item,int max_n){

    printf("GENERATING RANDOM BUFFER\n");
    int size = cuda_item->buffer_size;
    int test_buffer[size];
    int number;
    for(int i = 0;i<size;i++){
        number = rand()%max_n;
        test_buffer[i] = number;
        cuda_item->src_buffer[i] = number;
    }
    BuffersPrint(&test_buffer,cuda_item->src_buffer);
}
