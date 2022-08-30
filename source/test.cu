#include <stdlib.h>
#include <stdio.h>

#include <cstdlib>
#include <iostream>

#include "src/common.h"
#include "src/utils.h"

using namespace nvcuda;

#define NUM_BLOCKS 4

__device__ int* dataptr[NUM_BLOCKS]; // Per-block pointer

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

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


void initialize_multiplicand(signed char *h_multiplicand)
{
    h_multiplicand[0] = 1;
    h_multiplicand[1] = 2;
    h_multiplicand[2] = 4;
    h_multiplicand[3] = 8;
    h_multiplicand[4] = 16;
    h_multiplicand[5] = 32;
    h_multiplicand[6] = 64;
    h_multiplicand[7] = -128;
    h_multiplicand[24] = 1;
    h_multiplicand[25] = 2;
    h_multiplicand[26] = 4;
    h_multiplicand[27] = 8;
    h_multiplicand[28] = 16;
    h_multiplicand[29] = 32;
    h_multiplicand[30] = 64;
    h_multiplicand[31] = -128;

}

void initialize_diag_multiplicand(half *h_multiplicand)
{
    printf("Stop it. Get some help1.\n");
    for (int i = 0 ; i < 8*16; i++)
    {
        h_multiplicand[i] = 0;
    }
    printf("Stop it. Get some help2.\n");
    h_multiplicand[0] = 1;
    h_multiplicand[17] = 1;
    h_multiplicand[17*2] = 1;
    h_multiplicand[17*3] = 1;
    h_multiplicand[17*4] = 1;
    h_multiplicand[17*5] = 1;
    h_multiplicand[17*6] = 1;
    h_multiplicand[17*7] = 1;

}

__global__ void wmma_test_v2(half *A, half *B, signed char *multiplicand)
{
    int warp_id = threadIdx.x / 32;

    // Declare the fragments
    nvcuda::wmma::fragment<wmma::matrix_a, 8, 32, 16, signed char, wmma::row_major> A_frag;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 32, 16, signed char, wmma::row_major> B_frag[8];
    nvcuda::wmma::fragment<wmma::accumulator, 8, 32, 16, int> C_frag[8];

    nvcuda::wmma::load_matrix_sync(A_frag, multiplicand, 16);

}

__global__ void wmma_test(half *A, half *B, float *C)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // for (int i = 0; i < 8; i++)
    // {
    //     for (int j = 0; j < 16; j++)
    //     {
    //         // if (i == 0 && j == 8)
    //         // {
    //         //     A[i*16+j] = 1;   
    //         // }
    //         // else{
    //         //     A[i*16+j] = 0;
    //         // }
    //         A[i*16+j] = i*16+j;
    //     }
    // }

    // for (int n = 0; n < 8; n++)
    // {
    //     for (int i = 0; i < 16; i++)
    //     {
    //         for (int j = 0; j < 32; j++)
    //         {
    //             B[n*16*32 + i*32 + j] = n;
    //         }
    //     }
    // }
    // Declare the fragments
    nvcuda::wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> A_frag;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> B_frag[8];
    nvcuda::wmma::fragment<wmma::accumulator, 8, 32, 16, float> C_frag[8];

    // Initialize the output to zero
    for (int i = 0; i < 8; i++)
    {
        nvcuda::wmma::fill_fragment(C_frag[i], 0);
    }

    // Load the inputs
    nvcuda::wmma::load_matrix_sync(A_frag, A, 16);
    // nvcuda::wmma::fill_fragment(A_frag, 0);
    // for (int i = 0; i < 8; i++)
    // {
    //     nvcuda::wmma::load_matrix_sync(B_frag[i], &B[i*16*32], 32);
    // }

    // if (threadIdx.x == 4)
    // {
    //     B_frag[0].x[0] = 1;
    // }

    #pragma unroll
    for (int k = 0; k < 16; k++)
    {
        int row = lane_id % 4 * 2 + k%2 + k%8/4*8;
        // int col = lane_id / 4 + (k/2)/4*2+(k%2);
        int col = lane_id / 4 + 8 * ((k/8)*2+(k%4)/2);
        int index = (k%8)/4;
        B_frag[warp_id].x[k] = col;
    }

    __syncthreads();

    // for (int i = 0; i < 8; i++)
    // {
    //     // Perform the matrix multiplication
    //     nvcuda::wmma::mma_sync(C_frag[i], A_frag, B_frag[i], C_frag[i]);

    //     // Store the output
    //     nvcuda::wmma::store_matrix_sync(&C[i*8*32], C_frag[i], 32, wmma::mem_row_major);
    // }

    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(C_frag[warp_id], A_frag, B_frag[warp_id], C_frag[warp_id]);

    // Store the output
    nvcuda::wmma::store_matrix_sync(&C[warp_id*8*32], C_frag[warp_id], 32, wmma::mem_row_major);

    if (threadIdx.x == 0)
    {
        int tmp = 7;
        int k = tmp % 8 / 4 * 8;
        printf("k = %d\n", k);
    }

}

__global__ void test_kernel(signed char *dA, signed char *dB)
{
    // printf("11111");
    __shared__ signed char smem1[32];
    __shared__ signed char smem2[32];

    smem1[threadIdx.x] = dA[threadIdx.x];
    dB[threadIdx.x] = smem1[threadIdx.x];

    // printf("smem");
    if (threadIdx.x == 0)
    {
        printf("smem1: %d\n", smem1[2]);
        printf("dB: %d\n", dB[2]);
    }
    // float ldg_reg[4];
    // ldg32_nc(ldg_reg)
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

    
    half *h_A = (half*)malloc(8 * 16 * sizeof(half));
    initialize_diag_multiplicand(h_A);
    cudaMemcpy(d_A, h_A, 8 * 16 * sizeof(half), cudaMemcpyHostToDevice);

    printf("Stop it. Get some help.\n");

    wmma_test<<<1, 256>>>(d_A, d_B, d_C);

    half *h_B = (half*)malloc(8 * 16 * 32 * sizeof(half));
    float *h_C = (float*)malloc(8 * 8 * 32 * sizeof(float));

    // cudaMemcpy(h_A, d_A, 8 * 16 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, 8 * 16 * 32 * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, 8 * 8 * 32 * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix(8, 16, h_A, "h_A");

    printf("-----------------\n");

    printMatrix(16, 32, h_B, "h_B");

    printf("-----------------\n");

    printMatrix(8, 32, h_C, "h_C", 6);




    // signed char *d_A, *d_B;
    // signed char *h_B = (signed char *)malloc(32 * sizeof(signed char));
    // signed char h_A[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
    // cudaMalloc((void**) &d_A, 32 * sizeof(signed char));
    // cudaMalloc((void**) &d_B, 32 * sizeof(signed char));
    // cudaMemcpy(d_A, h_A, 32*sizeof(signed char), cudaMemcpyHostToDevice);

    // printf("test begin\n");
    // test_kernel<<<1, 32>>>(d_A, d_B);
    // cudaMemcpy(h_B, d_B, 32*sizeof(signed char), cudaMemcpyDeviceToHost);

    // cudaDeviceSynchronize();
    // printf("test end\n");

    // for (int i = 0; i < 32; i++)
    // {
    //     printf("value: %d\n", h_B[i]);
    // }

    // std::cout << std::endl;

    return 0;
}