/*
 * mm.h
 *
 *  Created on: Mar 26, 2019
 *      Author: ore
 */

#ifndef SRC_MM_H_
#define SRC_MM_H_


#include <stddef.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <cusp/csr_matrix.h>

#define HALF_MAX 65520.f
#define HALF_MIN 0.00000006f

#define BMP_DIM 8U //The dimension of the bitmap block

#define WARPS_BLOCK 2 //warps per block
#define TILES_WARP 2 //tiles per warp
#define TILES_BLOCK (WARPS_BLOCK * TILES_WARP) //tiles per block

#define DEBUG 0 // Print additional info from the algo
#define DEBUG_API 0 // Collect CUDA API errors
#define TIMING_VERBOSE 0 // Time specific algorithm steps

#define GPU_WARMUP 1 // Make GPU come out of low power mode
// How many times the implementation will be repeated in order to get average of results.
// 1 repetition is gives faster time, something with memory allocation/deallocation?
// Also CUSP has memory problems with more than 1 repetitions
#define REPETITIONS 1


#define gpuErrchk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char const *const source, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s: %s %s %d: %s\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line,
                source);
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
}


using IndexTypeH = int;  //indices
using ElemIndexTypeH = uint32_t;
using ValueTypeBMPH = thrust::tuple<uint32_t, uint64_t>; //index in values array, bitmap
using ValueTypeBMPH_NT = uint64_t; //bitmap
using ValueTypeH = signed char; // the actual values of values array

// Added by Shulai
using InitValueType = float;
using ValueType = signed char;
using OutputType = float;
using BitMaskType = int;

using MatrixTypeH = cusp::coo_matrix<IndexTypeH,ValueTypeBMPH,cusp::device_memory>;
using MatrixTypeH_NT = cusp::coo_matrix<IndexTypeH,ValueTypeBMPH_NT,cusp::device_memory>;

using MatrixTypeCOO = cusp::coo_matrix<IndexTypeH,ValueTypeH,cusp::host_memory>;

void multiplyBmp(const MatrixTypeH& A, const thrust::device_vector<ValueTypeH>& A_elems, const MatrixTypeH& B,
    const thrust::device_vector<ValueTypeH>& B_elems, MatrixTypeH& C, thrust::device_vector<ValueTypeH>& C_elems);

// template <typename ValueType>
void multiplyBmp_noTuple(const MatrixTypeH_NT& A, const thrust::device_vector<signed char>& A_elems,
        const thrust::device_vector<ElemIndexTypeH>& A_idx, const MatrixTypeH_NT& B,
        const thrust::device_vector<signed char>& B_elems, const thrust::device_vector<ElemIndexTypeH>& B_idx,
        MatrixTypeH_NT& C, thrust::device_vector<signed char>& C_elems, thrust::device_vector<ElemIndexTypeH>& C_idx);


void get_characteristics(const MatrixTypeH& A, const thrust::device_vector<ValueTypeH>& A_elems, const MatrixTypeH& B,
    const thrust::device_vector<ValueTypeH>& B_elems, MatrixTypeH& C, thrust::device_vector<ValueTypeH>& C_elems,
    const MatrixTypeCOO& A_coo, const MatrixTypeCOO& B_coo, MatrixTypeCOO& C_coo);


struct find_tile_index
{
    using IndexType = int;
    using LongIndexType = uint64_t;
    using BMPType = uint64_t;

    IndexType ncols_;

    find_tile_index(IndexType num_cols) {
        ncols_ = num_cols / BMP_DIM + ((num_cols % BMP_DIM)?1:0) ;
    }

    __host__ __device__
    void operator()(thrust::tuple<IndexType &, IndexType &, LongIndexType &, BMPType &> x) {
        LongIndexType ncols = ncols_; //promotes result to uint64_t, necessary for bigger matrices
        // Absolute index of the tile this element belongs to
        x.get<2>() = (x.get<0>() / BMP_DIM) * ncols + x.get<1>() / BMP_DIM;

        //Absolute index of element inside its tile
        x.get<3>() = 1ULL << ( (x.get<0>() % BMP_DIM) * BMP_DIM + x.get<1>() % BMP_DIM );
    }
};

struct absolute2relative
{
    using IndexType = int;
    using LongIndexType = uint64_t;

    IndexType ncols_;

    absolute2relative(IndexType num_cols): ncols_(num_cols) {}

    __host__ __device__
    void operator()(thrust::tuple<LongIndexType &, IndexType &, IndexType &> x) {
        LongIndexType ncols = ncols_; //promotes result to uint64_t, necessary for bigger matrices
        x.get<1>() = x.get<0>() / ncols;
        x.get<2>() = x.get<0>() - x.get<1>() * ncols; //modulo
    }
};

struct absolute
{
    using ValueType = float;

  __host__ __device__
  void operator()(ValueType &x)
  {
      x = fabsf(x);
  }
};

struct bmp_popcount_d
{
    using UnsignedIndexType = uint32_t;
    using BMPType = uint64_t;

  __device__
  UnsignedIndexType operator()(BMPType rhs)
  {
      return (UnsignedIndexType) __popcll(rhs); //TODO not GPU compatible
  }
};


#endif /* SRC_MM_H_ */
