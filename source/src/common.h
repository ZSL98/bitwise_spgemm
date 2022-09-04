#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <bitset>

#include <cusparse.h>
#include <mma.h>

using namespace nvcuda;

#define REPETITIONS 1

#define SPARSITY 99
#define SPARSITY_A 99
#define SPARSITY_B 99
#define BIT_WIDTH 8
#define MAX_GROUP_NUM 4
#define OUTPUT_MAX_GROUP_NUM 16
#define TILE_HEIGHT 256
#define TILE_WIDTH 32
#define SIZE_M 4096
#define SIZE_K 4096
#define SIZE_N 4096
#define SPLIT_K 256

#define MAX_TILEA_NNZ 2048
#define MAX_SPILLED_ROW_CNT_C 1024
#define MAX_LINE_NNZ_A 4

#define PRINT_MAT_A_INFO false
#define PRINT_MAT_B_INFO false

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

