#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "cublas_v2.h"
#include <inttypes.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void printintMatrix(int m, int n, const int*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            int Areg = A[col + row*n];
            printf("%s(%d,%d) = %d\n", name, row+1, col+1, Areg);
        }
    }
}

void printfloatMatrix(int m, int n, const float*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[col + row*n];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void fill_random(float*data, int m, int n, int sparsity) {
	for (int i = 0; i < m*n; i++) {
		data[i] = rand() % 100;
		if (data[i] < sparsity) //made sparse 
			data[i] = 0;
	}
}

void fill_random_uint8(uint8_t*data, int m, int n, int sparsity) {
	for (int i = 0; i < m*n; i++) {
		data[i] = rand() % 100;
		if (data[i] < sparsity) //made sparse 
			data[i] = 0;
	}
}

__global__ void layout_trans(float* src, int* dst, size_t row, size_t col, int bit_width) {
    // int bid = blockIdx.y * gridDim.x + blockIdx.x;
	// int tid = bid * blockDim.x + threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t MatSize = row * col;
    uint32_t *value = (uint32_t *)&src[tid]; 
    for(int i = 0; i < bit_width; i++)
    {
        dst[i*MatSize+tid] = ((*value >> i) & 1);
    }
}

__global__ void uint8_layout_trans(uint8_t* src, int* dst, size_t row, size_t col) {
    // int bid = blockIdx.y * gridDim.x + blockIdx.x;
	// int tid = bid * blockDim.x + threadIdx.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t MatSize = row * col;
    uint8_t *value = (uint8_t *)&src[tid]; 
    for(int i = 0; i < 8; i++)
    {
        dst[i*MatSize+tid] = ((*value >> i) & 1);
    }
}


__global__ void layout_compact(int* src, float* dst, size_t row, int bit_width){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int add_times[1024];
    for (int b = 0; b < bit_width; b++) {
        add_times[threadIdx.x] = src[tid];
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            // Each thread does work unless it is further than the stride
            if (threadIdx.x < s) {
                add_times[threadIdx.x] = add_times[threadIdx.x] + add_times[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            dst[b*row+blockIdx.x] = add_times[0];
        }
    }
}

__global__ void indexed_spmm_1d(int* mask,
                                float* matB, 
                                float* matC, 
                                size_t m, 
                                size_t n, 
                                size_t k, 
                                int bit_width){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t MatASize = m * k;
    for (int b = 0; b < bit_width; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                if (mask[b*MatASize+i*k+j] == 1){
                    matC[tid] += matB[j*n+threadIdx.x];
                }
            }
        }
    }

    // for (int i = 0; i < sum(index[threadIdx.y]); i++){
    //     matC[tid] += matB[index[threadIdx.y][i]*blockDim.x+threadIdx.x];
    // }
}

__global__ void indexed_spmm_2d(int* mask, float* matB, float* matC, int bit_width){
    // gridDim.x = k  -> pointer: blockIdx.x
    // gridDim.y = m  -> pointer: blockIdx.y
    // blockDim.x = n -> pointer: threadIdx.x
    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int tid = bid * blockDim.x + threadIdx.x; 
    for (int b = 0; b < bit_width; b++) {
        if (mask[b*gridDim.x*gridDim.y+bid] == 1){
            matC[tid] += matB[blockIdx.x*blockDim.x+threadIdx.x] * (1<<b);
        }
    }
}

__global__ void uint8_spmm_2d_1d(int* mask, float* matB, float* matC){
    // gridDim.x = k  -> pointer: blockIdx.x
    // gridDim.y = m  -> pointer: blockIdx.y
    // blockDim.x = n -> pointer: threadIdx.x
    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int tid = bid * blockDim.x + threadIdx.x; 
    for (int b = 0; b < 8; b++) {
        if (mask[b*gridDim.x*gridDim.y+bid] == 1){
            float value = matB[blockIdx.x*blockDim.x+threadIdx.x] * (1<<b);
            atomicAdd( &matC[blockIdx.y*blockDim.x+threadIdx.x], value);
        }
    }
    __threadfence();
}

__global__ void uint8_spmm_1d_2d(int* mask, float* matB, float* matC){
    // gridDim.x = k  -> pointer: blockIdx.x
    // gridDim.y = m  -> pointer: blockIdx.y
    // blockDim.x = n -> pointer: threadIdx.x
    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int tid = bid * blockDim.x + threadIdx.x; 
    for (int b = 0; b < 1; b++) {
        if (mask[b*gridDim.x*gridDim.y+bid] == 1){
            matC[blockIdx.y*blockDim.x+threadIdx.x] += matB[blockIdx.x*blockDim.x+threadIdx.x] * (1<<b);
            printf("%d, %d, %d, %f, %f\n", bid, blockIdx.y*blockDim.x+threadIdx.x, blockIdx.x*blockDim.x+threadIdx.x, matC[blockIdx.y*blockDim.x+threadIdx.x], matB[blockIdx.x*blockDim.x+threadIdx.x]);
        }
    }
    __syncthreads();
}


void uint8_dense2sparse_csr(const uint8_t* h_dense, int num_rows, int num_cols, cusparseSpMatDescr_t &SpMat){
    int dense_size = num_rows * num_cols;
    // Device memory management
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values,  *d_dense;
    cudaMalloc((void**) &d_dense, dense_size * sizeof(float));
    cudaMalloc((void**) &d_csr_offsets, (num_rows + 1) * sizeof(int));
    cudaMemcpy(d_dense, h_dense, dense_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create dense matrix A
    cusparseCreateDnMat(&matA, num_rows, num_cols, num_cols, d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create sparse matrix B in CSR format
    cusparseCreateCsr(&SpMat, num_rows, num_cols, 0,
                        d_csr_offsets, NULL, NULL,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // allocate an external buffer if needed
    cusparseDenseToSparse_bufferSize(
                        handle, matA, SpMat,
                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                        &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_analysis(handle, matA, SpMat,
                                    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                    dBuffer);
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    cusparseSpMatGetSize(SpMat, &num_rows_tmp, &num_cols_tmp, &nnz);

    // allocate CSR column indices and values
    cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int));
    cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float));
    // reset offsets, column indices, and values pointers
    cusparseCsrSetPointers(SpMat, d_csr_offsets, d_csr_columns, d_csr_values);
    // execute Sparse to Dense conversion
    cusparseDenseToSparse_convert(handle, matA, SpMat,
                                CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer);
    
}

void run_cusparseSpMM(float* dA_dense, float *dB, float *dC, int m, int k, int n)
{
    // float hA_dense[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // cudaMemcpy(hA_dense, dA_dense, 16*sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i<16;i++){
    //     printf("%d", hA_dense[i]);
    // }

    float alpha = 1.0f;
    float beta = 0.0f;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseHandle_t     handle = NULL;
    cudaEvent_t          start, stop;

    cusparseCreate(&handle);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Device memory management
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values;
    cudaMalloc((void**) &d_csr_offsets, (m + 1) * sizeof(int));

    cusparseDnMatDescr_t matA;
    cusparseSpMatDescr_t SpMat;
    cusparseDnMatDescr_t matB, matC;
    // Create dense matrix A
    cusparseCreateDnMat(&matA, m, k, k, dA_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&SpMat, m, k, 0,
                        d_csr_offsets, NULL, NULL,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // allocate an external buffer if needed
    cusparseDenseToSparse_bufferSize(
                        handle, matA, SpMat,
                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                        &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    // printf("Buffersize %d", bufferSize);

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_analysis(handle, matA, SpMat,
                                    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                    dBuffer);
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    cusparseSpMatGetSize(SpMat, &num_rows_tmp, &num_cols_tmp, &nnz);
    printf("Number of non-zeros: %" PRId64 "\n", nnz);

    // allocate CSR column indices and values
    cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int));
    cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float));
    // reset offsets, column indices, and values pointers
    cusparseCsrSetPointers(SpMat, d_csr_offsets, d_csr_columns, d_csr_values);
    // execute Sparse to Dense conversion
    cusparseDenseToSparse_convert(handle, matA, SpMat,
                                CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer);


    // Create dense matrix B
    cusparseCreateDnMat(&matB, k, n, k, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, m, n, m, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, SpMat, matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMM
    cudaEventRecord(start);

    cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, SpMat, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cusparse execution time: %f \n", milliseconds);

}

int main(int argc, char*argv[]) {

    // cudaStream_t         stream = NULL;
    cudaEvent_t          start, stop;

    // CHECK_CUDA( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) )
    CHECK_CUDA( cudaEventCreate(&start) )
    CHECK_CUDA( cudaEventCreate(&stop) )

	const int m = 16;
    const int k = 1024;
	const int n = 16;
    int bit_width = 8;

    // Tiny matrix
	// const uint8_t hA[m*k] = {1, 0, 2, 3, 0, 4, 0, 0, 5, 0, 6, 7, 0, 8, 0, 9};
    // const float hA_float[m*k] = {1, 0, 2, 3, 0, 4, 0, 0, 5, 0, 6, 7, 0, 8, 0, 9};
    // const float hB[k*n] = {0, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9};
    // float hC[m*n] = {1, 0, 5, 0, 0, 4, 0, 8, 2, 0, 6, 0, -3, 0, 7, 9};

    int *hmask = (int*)malloc(bit_width * m * k * sizeof(int));;

    // Huge matrix
    int sparsity = 0;
	float* hA_float = (float*)malloc(sizeof(float)*m*k);
    uint8_t* hA = (uint8_t*)malloc(sizeof(uint8_t)*m*k);
	fill_random(hA_float, m, k, sparsity);
    fill_random_uint8(hA, m, k, sparsity);
    float* hB = (float*)malloc(sizeof(float)*k*n);
    float* hC = (float*)malloc(sizeof(float)*m*n);
    fill_random(hB, k, n, sparsity);
    fill_random(hC, m, n, sparsity);

    uint8_t *dA;
    float *dA_float;
    float *dB, *dC;
    int *dmask;

    // CHECK_CUDA( cudaMalloc((void**) &dA,         A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dA,        m * k * sizeof(uint8_t)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_float,  m * k * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB,        k * n * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,        m * n * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dmask,  bit_width * m * k * sizeof(int)) )
    
    CHECK_CUDA( cudaMemcpy(dA, hA, m * k * sizeof(uint8_t),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_float, hA_float, m * k * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, k * n * sizeof(float),
                           cudaMemcpyHostToDevice) )

    // cusparse implementation
    run_cusparseSpMM(dA_float, dB, dC, m, k, n);
    CHECK_CUDA( cudaMemcpy(hC, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost) )
    // printfloatMatrix(m, n, hC, "MatC");

    // layout_trans<<<m, k>>>(dA, dmask, m, k, bit_width);
    uint8_layout_trans<<<m, k>>>(dA, dmask, m, k);
    CHECK_CUDA( cudaMemcpy(hmask, dmask, bit_width * m * k * sizeof(int),
                           cudaMemcpyDeviceToHost) )

    // printintMatrix(bit_width*m, k, hmask, "Mask");

    // layout_compact<<<grid, block>>>(tmp_mask, index);
    CHECK_CUDA( cudaMemset(dC, 0, m * n * sizeof(float)) )
    CHECK_CUDA( cudaEventRecord(start) )

    dim3 grid(m, k, 1), block(n, 1, 1);
    // indexed_spmm_1d<<<m, n>>>(dmask, dB, dC, m, n, k, bit_width);
    // indexed_spmm_2d<<<grid, block>>>(dmask, dB, dC, bit_width);
    uint8_spmm_2d_1d<<<grid, block>>>(dmask, dB, dC);

    CHECK_CUDA( cudaEventRecord(stop) )
    CHECK_CUDA( cudaEventSynchronize(stop) )
    float milliseconds = 0;
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, start, stop) )
    printf("Execution Time: %f \n", milliseconds);

    CHECK_CUDA( cudaMemcpy(hC, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost) )
    // printfloatMatrix(m, n, hC, "MatC");

}
