#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[j + p * n];
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}


__global__ void bit_wise_gemm_s(int M, int N, int K, int* A_ctl_seq, float* matB, float* matC){
    // gridDim.x = k  -> pointer: blockIdx.x
    // gridDim.y = m  -> pointer: blockIdx.y
    // blockDim.x = n -> pointer: threadIdx.x

    // gris: 2 dimensions, block: 2 dimensions
    // int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
    // int threadId = blockId * (blockDim.x * blockDim.y)  
    //                    + (threadIdx.y * blockDim.x) + threadIdx.x;

    // grid: 3 dimensions, block: 1 dimension
    // int blockId = blockIdx.x + blockIdx.y * gridDim.x  
    //                  + gridDim.x * gridDim.y * blockIdx.z;  
    // int threadId = blockId * blockDim.x + threadIdx.x;

    // int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    // int tid = bid * blockDim.x + threadIdx.x;

    float inputs[16];    // 2^k = 2^4 = 16
    float outputs[64];   // m = 32
    //int src_reg_idx;     // the index of which register's data to add on
    int dst_reg_idx;     // the index of which register to store the added result
    int shift_cnt;       // the number of shifting bits
    int output_reg_idx;  // the index of which register to store the shifted value
    __shared__ int A_smem_ctl_seq[64 * 32];    // m*b
    __shared__ float B_smem_tile[4 * 64];  // k*n

    // load the 1'st A tiled control sequence and B tile to smem from gmem
    // This is a naive version. Use inline PTX to promote
    # pragma unroll
    for (int i=0; i<64*32; i++){
        A_smem_ctl_seq[i] = A_ctl_seq[i];
    }
    # pragma unroll
    for (int i=0; i<4*64; i++){
        B_smem_tile[i] = matB[i];
    }

    // K-loop
    // for (int k=0; k<K; k+=4){
    __syncthreads();

    inputs[0] = B_smem_tile[threadIdx.x];
    inputs[1] = B_smem_tile[64 + threadIdx.x];
    inputs[2] = B_smem_tile[128 + threadIdx.x];
    inputs[3] = B_smem_tile[192 + threadIdx.x];
    inputs[4] = inputs[0] + inputs[1];
    inputs[5] = inputs[0] + inputs[2];
    inputs[6] = inputs[0] + inputs[3];
    inputs[7] = inputs[1] + inputs[2];
    inputs[8] = inputs[1] + inputs[3];
    inputs[9] = inputs[2] + inputs[3];
    inputs[10] = inputs[4] + inputs[2];
    inputs[11] = inputs[4] + inputs[3];
    inputs[12] = inputs[5] + inputs[3];
    inputs[13] = inputs[7] + inputs[3];
    inputs[14] = inputs[10] + inputs[3];

    # pragma unroll
    for (int i=0; i<64*32; i++){
        dst_reg_idx    = A_smem_ctl_seq[i];
        // shift_cnt      = A_smem_ctl_seq[3 * i + 1];
        // output_reg_idx = A_smem_ctl_seq[3 * i + 2];
        outputs[i/32] += (__float_as_int(inputs[dst_reg_idx]) << i%32);
        // outputs[output_reg_idx] += inputs[dst_reg_idx];
    }

        // if(k + 4 < K){
            // # pragma unroll
            // for (int i=0; i<64*32*3; i++){
            //     A_smem_ctl_seq[i] = A_ctl_seq[i];
            // }
            // # pragma unroll
            // for (int i=0; i<4*64; i++){
            //     B_smem_tile[i] = matB[i];
            // }
        // }
    // }

    //load m*n results to gmem, m = 32
    #pragma unroll
    for (int i=0; i<64; i++){
        matC[(blockIdx.y * 64 + i) * N + blockIdx.x * 64 + threadIdx.x] += outputs[i];
    }

}

int main() {
    int m = 5120;
    int n = 4096;
    int k = 4096;
    int n_iter = 10;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, m * k * sizeof(float));
    cudaMallocHost(&h_B, k * n * sizeof(float));
    cudaMallocHost(&h_C, m * n * sizeof(float));
    random_init(h_A, m * k);
    random_init(h_B, k * n);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid((n + 63) / 64, (m + 63) / 64, (k + 3) / 4);
    dim3 block(64);

    int *h_A_ctl, *d_A_ctl;
    cudaMallocHost(&h_A_ctl, 64 * 32 * 3 * sizeof(int));
    cudaMemset(&h_A_ctl, 0, 64 * 32 * 3);
    cudaMalloc(&d_A_ctl, 64 * 32 * 3 * sizeof(int));
    cudaMemcpy(d_A_ctl, h_A_ctl, 64 * 32 * 3 * sizeof(int), cudaMemcpyDefault);

    // warmup
    bit_wise_gemm_s<<<grid, block>>>(
        m, n, k, d_A_ctl, d_B, d_C);

    cudaEventRecord(start);
    for (int i = 0; i < n_iter; ++i) {
        bit_wise_gemm_s<<<grid, block>>>(
            m, n, k, d_A_ctl, d_B, d_C);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    long workload = n_iter * long(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDefault);

    bool chk = check(h_A, h_B, h_C, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
}
