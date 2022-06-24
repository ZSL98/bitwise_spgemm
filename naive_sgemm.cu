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
