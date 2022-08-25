#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_
#include "common.h"


template <typename InitValueType,
          typename ValueType>
__global__ void format_convert(InitValueType *d_dense_in, ValueType *d_dense_out)
{
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int entry = row * blockDim.x * gridDim.x + col;
    d_dense_out[entry] = (ValueType)d_dense_in[entry];
}


#endif