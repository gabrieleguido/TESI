#define BLOCK_DIM 1024
#define DEBUG 0
#define SLEEPING_TIME 1
#define RANDOM_RANGE 100

typedef struct MyCudaItem MyCudaItem;
typedef struct ThreadArgs ThreadArgs;

typedef void*(*KernelLaunchFn)(void*);

struct MyCudaItem{
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

void BufferPrint(int* buff,int n, const char* name);

void MyCudaReduction(MyCudaItem* cuda_item);

void* MyCudaReductionKernel(void* args);

int MyCudaIterations(MyCudaItem* cuda_item);

void MyCudaSetThreadArgs(ThreadArgs* targs,int threadIdx, int blockIdx, int* dest, int* src, int size);

void MyCudaKernelLaunch(MyCudaItem* cuda_item);

int MyCudaIsPowerTwo(int n);

void MyCudaPrefixSum(MyCudaItem* cuda_item);

void* MyCudaPrefixDownPassKernel(void* args);

void* MyCudaPrefixUpPassKernel(void* args);

void MyCudaPrefixSumTest(MyCudaItem* cuda_item,int max_n);



