#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <bitset>

#include <cusparse.h>

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE float
#endif

#define SPARSITY 95
#define BIT_WIDTH 8
#define MAX_GROUP_NUM 8
#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define SIZE_M 1024
#define SIZE_K 1024
#define SIZE_N 1024
#define SPLIT_K 256
#define MAX_TILEA_NNZ 1024

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

