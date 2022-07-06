#include <stdlib.h>
#include <stdio.h>

#define NUM_BLOCKS 4

__device__ int* dataptr[NUM_BLOCKS]; // Per-block pointer

__global__ void allocmem()
{
    // Only the first thread in the block does the allocation
    // since we want only one allocation per block.
    if (threadIdx.x == 0)
        dataptr[blockIdx.x] = (int*)malloc(blockDim.x * 4);
    __syncthreads();

    // Check for failure
    if (dataptr[blockIdx.x] == NULL)
        return;

    // Zero the data with all threads in parallel
    dataptr[blockIdx.x][threadIdx.x] = 0;
}

// Simple example: store thread ID into each element
__global__ void usemem()
{
    int* ptr = dataptr[blockIdx.x];
    if (ptr != NULL)
        ptr[threadIdx.x] += threadIdx.x;
}

// Print the content of the buffer before freeing it
__global__ void freemem()
{
    int* ptr = dataptr[blockIdx.x];
    if (ptr != NULL)
        printf("Block %d, Thread %d: final value = %d\n",
                      blockIdx.x, threadIdx.x, ptr[threadIdx.x]);

    // Only free from one thread!
    if (threadIdx.x == 0)
        free(ptr);
}

__global__ void my_kernel(int *d_mem1, int **d_mem2)
{
    printf("dataptr1: %d\n", &dataptr[0]);
    printf("dataptr2: %d\n", dataptr[0]);
    printf("dataptr3: %d\n", &dataptr[0][0]);
    printf("dataptr4: %d\n", dataptr[0][0]);
    printf("dataptr5: %d\n", *(dataptr[0]+1));

    printf("d_mem2: %d\n", d_mem2[0]);
    d_mem2 = &dataptr[0];
    printf("d_mem1: %p\n", d_mem1);
    printf("d_mem1: %d\n", d_mem1[0]);
    printf("d_mem1: %p\n", &d_mem1);
    printf("d_mem1: %d\n", &d_mem1[0]);
    printf("d_mem2: %p\n", &d_mem2);
    printf("d_mem2: %d\n", *d_mem2[0]);
    printf("d_mem2: %d\n", *d_mem2[1]);
    __syncthreads();

}

__global__ void my_kernel2(int *d_mem1, int *d_mem2)
{

    d_mem2[0] = 99;
    __syncthreads();
}

int main()
{
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    // Allocate memory
    allocmem<<< NUM_BLOCKS, 10 >>>();

    // Use memory
    usemem<<< NUM_BLOCKS, 10 >>>();

    int **h_mem = (int**)malloc(sizeof(int*));
    int *h_mem_ = (int*)malloc(10*sizeof(int));
    int **d_mem1, *d_mem2;
    cudaMalloc((void**)&d_mem1, sizeof(int*));
    // cudaMalloc((void**)&d_mem2, 10 * sizeof(int));
    printf("initial d_mem2: %p\n", d_mem2);
    cudaGetSymbolAddress((void**)&d_mem1, dataptr);

    cudaMemcpy(h_mem, d_mem1, sizeof(int*), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_mem_, h_mem[0], 10*sizeof(int), cudaMemcpyDeviceToHost);
    printf("h_mem: %p\n", h_mem[0]);
    // cudaMemcpy(h_mem[0], d_mem2, 10*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaGetSymbolAddress((void**)&d_mem2, d_mem1[0]);
    // cudaMemcpy(h_mem, d_mem, 10*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpyFromSymbol(d_mem2, dataptr[0], 10*sizeof(int), 0);
    // cudaMemcpyToSymbol(dataptr[0], d_mem, sizeof(int*));

    // my_kernel<<<1, 1>>>(d_mem1, &d_mem2);

    // my_kernel2<<<1, 1>>>(d_mem1, d_mem2);

    // cudaDeviceSynchronize();
    // printf("modified d_mem2: %p\n", d_mem2);
    // cudaMemcpy(h_mem, d_mem2, 10*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpyFromSymbol(h_mem, &dataptr[0], 10*sizeof(int), 0);

    for (int i = 0; i < 10; i++)
    {
        printf("hB_tile_spilled_csrColInd--%d: %d\n", i, h_mem_[i]);
    }
    // usemem<<< NUM_BLOCKS, 10 >>>();
    // usemem<<< NUM_BLOCKS, 10 >>>();

    // Free memory
    freemem<<< NUM_BLOCKS, 10 >>>();

    cudaDeviceSynchronize();

    return 0;
}