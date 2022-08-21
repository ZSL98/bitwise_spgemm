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

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE float
#endif

#define SPARSITY 99
#define SPARSITY_A 99
#define SPARSITY_B 99.8
#define BIT_WIDTH 8
#define MAX_GROUP_NUM 4
#define OUTPUT_MAX_GROUP_NUM 16
#define TILE_HEIGHT 256
#define TILE_WIDTH 32
#define SIZE_M 2048
#define SIZE_K 2048
#define SIZE_N 2048
#define SPLIT_K 256

#define MAX_TILEA_NNZ 2048
#define MAX_SPILLED_ROW_CNT_C 1024
#define MAX_LINE_NNZ_A 4

#define PRINT_MAT_A_INFO false
#define PRINT_MAT_B_INFO false

#ifndef SMATRIX
#define SMATRIX
typedef struct 
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
    int tilem;
    int tilen;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;
    MAT_VAL_TYPE *tile_csr_Value;
    unsigned char *tile_csr_Col;
    unsigned char *tile_csr_Ptr;
    unsigned short *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
}SMatrix;
#endif

#ifndef CSRMATRIX
#define CSRMATRIX
typedef struct 
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
    int tilem;
    int tilen;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;
    MAT_VAL_TYPE *tile_csr_Value;
    unsigned char *tile_csr_Col;
    unsigned char *tile_csr_Ptr;
    unsigned short *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
}CSRMatrix;
#endif

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

