typedef struct MyCudaItem MyCudaItem;

typedef void(*KernelLaunchFn)(MyCudaItem*);

struct MyCudaItem{
    int n_threads;
    int n_blocks;
    int buffer_size;
    int* src_buffer;
    int* dest_buffer;
    KernelLaunchFn* kernel_fn;
};

void MyCudaMalloc(MyCudaItem* cuda_item, int n);

void MyCudaMemSwap(MyCudaItem* cuda_item);


void MyCudaMemFree(MyCudaItem* cuda_item);

void MyCudaBuffersPrint(MyCudaItem* cuda_item);




