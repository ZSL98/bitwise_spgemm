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

#define BIT_WIDTH 8
#define MAX_GROUP_NUM 4
#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define SIZE_M 64
#define SIZE_K 64
#define SIZE_N 64
#define SPLIT_K 64

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

void fill_random_uint8(uint8_t*data, int m, int n, int sparsity) {
	for (int i = 0; i < m*n; i++) {
		data[i] = rand() % 100;
		if (data[i] < sparsity) //made sparse 
			data[i] = 0;
	}
}

void fill_random(float*data, int m, int n, int sparsity) {
	for (int i = 0; i < m*n; i++) {
		data[i] = rand() % 100;
		if (data[i] < sparsity) //made sparse 
			data[i] = 0;
	}
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
            // printf("%s(%d,%d) = %d\n", name, row+1, col+1, Areg);
            std::cout << std::left << std::setw(4) << Areg;
        }
        std::cout << std::endl;
    }
}

void printfloatMatrix(int m, int n, const float*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[col + row*n];
            // printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
            std::cout << std::left << std::setw(4) << Areg;
        }
        std::cout << std::endl;
    }
}

void _itoa(const unsigned long long int a, char *s)
{
    for (int i = 0; i < 64; i++)
    {
        if (((a >> i) & 1) == 0x01)
        {
            s[i] = '1';
        }
        else {s[i] = '0';}
    }
}

void printlongintMatrix(int m, int n, const unsigned long long int*A, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            unsigned long long int Areg = A[col + row*n];
            char s[64];
            _itoa(Areg, s);
            // printf("%s(%d,%d) = %s\n", name, row+1, col+1, s);
            for (int i = 0; i < 64; i++) {
                std::cout << s[i];
            }
            std::cout << std::endl;
        }
    }
}