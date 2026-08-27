#define BLOCK_DIM 1024

typedef struct MyCudaItem MyCudaItem;
typedef struct ThreadArgs ThreadArgs;

typedef void*(*KernelLaunchFn)(void*);

struct MyCudaItem{
    // int n_threads;
    // int n_blocks;
    int buffer_size;
    int size;
    int* src_buffer;
    int* dest_buffer;
    KernelLaunchFn kernel_fn;
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

void BuffersPrint(int* src, int* dest, int size);

void MyCudaReduction(MyCudaItem* cuda_item);

void* MyCudaReductionKernel(void* args);

int MyCudaIterations(MyCudaItem* cuda_item);

void MyCudaSetThreadArgs(ThreadArgs* targs,int threadIdx, int blockIdx, int* dest, int* src, int size);

void MyCudaKernelLaunch(MyCudaItem* cuda_item);

void MyCudaPrefixSum(MyCudaItem* cuda_item);

void* MyCudaPrefixDownPassKernel(void* args);




