#define BLOCK_DIM 1024

typedef struct MyCudaItem MyCudaItem;
typedef struct ThreadArgs ThreadArgs;

typedef void(*KernelLaunchFn)(int,int,void*,const void*,int);

struct MyCudaItem{
    // int n_threads;
    // int n_blocks;
    int buffer_size;
    int* src_buffer;
    int* dest_buffer;
    KernelLaunchFn* kernel_fn;
};

struct ThreadArgs{
    int threadIdx;
    int blockIdx;
    void* dest;
    void* src;
    int size;
};

void MyCudaMalloc(MyCudaItem* cuda_item, int n);

void MyCudaMemSwap(MyCudaItem* cuda_item);

void MyCudaMemFree(MyCudaItem* cuda_item);

void MyCudaBuffersPrint(MyCudaItem* cuda_item);

void MyCudaReduction(MyCudaItem* cuda_item);

void* MyCudaReductionPass(void* args);

int MyCudaIterations(MyCudaItem* cuda_item);




