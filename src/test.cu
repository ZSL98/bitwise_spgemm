#include <stdlib.h>
#include <stdio.h>
#include "common.h"
#include "utils.h"

using namespace nvcuda;

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

__global__ void wmma_test(half *A, half *B, float *C)
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            A[i*16+j] = j%8;
        }
    }

    for (int n = 0; n < 8; n++)
    {
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 32; j++)
            {
                B[n*16*32 + i*32 + j] = j%8 + n;
            }
        }
    }
    // Declare the fragments
    nvcuda::wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> A_frag;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> B_frag[8];
    nvcuda::wmma::fragment<wmma::accumulator, 8, 32, 16, float> C_frag[8];

    // Initialize the output to zero
    for (int i = 0; i < 8; i++)
    {
        nvcuda::wmma::fill_fragment(C_frag[i], 0.0f);
    }

    // Load the inputs
    nvcuda::wmma::load_matrix_sync(A_frag, A, 16);
    for (int i = 0; i < 8; i++)
    {
        nvcuda::wmma::load_matrix_sync(B_frag[i], B, 32);
    }

    for (int i = 0; i < 8; i++)
    {
        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(C_frag[i], A_frag, B_frag[i], C_frag[i]);

        // Store the output
        nvcuda::wmma::store_matrix_sync(&C[i*8*32], C_frag[i], 32, wmma::mem_row_major);
    }
}

int main()
{
    /*
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

    cudaDeviceSynchronize();*/

    half *d_A, *d_B;
    float *d_C;

    CHECK_CUDA( cudaMalloc((void**) &d_A, 8 * 16 * sizeof(half))   )
    CHECK_CUDA( cudaMalloc((void**) &d_B, 8 * 16 * 32 * sizeof(half))   )
    CHECK_CUDA( cudaMalloc((void**) &d_C, 8 * 8 * 32 * sizeof(float))   )

    wmma_test<<<1, 256>>>(d_A, d_B, d_C);

    half *h_A = (half*)malloc(8 * 16 * sizeof(half));
    half *h_B = (half*)malloc(8 * 16 * 32 * sizeof(half));
    float *h_C = (float*)malloc(8 * 8 * 32 * sizeof(float));

    cudaMemcpy(h_A, d_A, 8 * 16 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, 8 * 16 * 32 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, 8 * 8 * 32 * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(8, 16, h_A, "h_A");

    printf("-----------------\n");

    printMatrix(16, 32, h_B+1*16*32, "h_B");

    printf("-----------------\n");

    printMatrix(16, 32, h_C, "h_C");

    return 0;
}