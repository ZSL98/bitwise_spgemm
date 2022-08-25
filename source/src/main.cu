#include <cuda_runtime_api.h>
#include <cusparse.h>

#include "common.h"
#include "transform.cuh"
#include "cuda_utils.cuh"
#include "utils.h"
#include "cusp/csr_matrix.h"
#include "cusp/timer.h"

// tsparse include
#include <thrust/host_vector.h>
#include <thrust/find.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/functional.h> //bit_or
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/omp/execution_policy.h>

#include <algorithm> //find
#include <vector>

#include "mm.h"

// include TileSpGEMM
// #include "TileSpGEMM/common.h"
// #include "TileSpGEMM/mmio_highlevel.h"
// #include "TileSpGEMM/utils.h"
// #include "TileSpGEMM/utils_cuda_scan.h"
// #include "TileSpGEMM/spgemm_nsparse_kernel.h"
// #include "TileSpGEMM/csr2tile.h"
#include "TileSpGEMM/tilespgemm-cuda.h"
// #include "TileSpGEMM/spgemm-cpu.h"
// #include "TileSpGEMM/tile2csr.h"
// #include "TileSpGEMM/spgemm_serialref_spa_new.h"
// #include "TileSpGEMM/spgemm_cu.h"



template <typename BitMaskType,
          typename InitValueType,
          typename ValueType>
__global__ void generate_groups(BitMaskType *MatB_bit,
                                BitMaskType *d_group_mask,
                                int *d_group_ele_row_ind,
                                ValueType *d_group_ele_row_val,
                                InitValueType *d_dense,
                                int *group_id,
                                int *spilled_row_cnt,
                                int *spilled_nnz,
                                // float **tile_spilled_csrVal,
                                // int **tile_spilled_csrColInd,
                                // int **tile_spilled_csrRowPtr,
                                int *spilled_row_hash_table_gmem,
                                int *spilled_row_hash_table_reverse_gmem
                                )
{
    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int row_ind = blockDim.x * blockIdx.y + threadIdx.x;
    int entry_ind = row_ind * gridDim.x * TILE_WIDTH + blockIdx.x * TILE_WIDTH;
    int entry_ind_bit = row_ind * gridDim.x + blockIdx.x;

    __shared__ int row_group[MAX_GROUP_NUM];
    __shared__ int group_ele_row_idx[MAX_GROUP_NUM][TILE_WIDTH];
    __shared__ InitValueType d_dense_smem[SPLIT_K][TILE_WIDTH];
    __shared__ int spilled_row_hash_table_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];
    // __shared__ int spilled_row_cnt[row_cnt/tile_height*col_cnt/tile_width];

    spilled_row_hash_table_smem[threadIdx.x] = 0;
    spilled_row_hash_table_reverse_smem[threadIdx.x] = -1;
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        d_dense_smem[threadIdx.x][i] = d_dense[entry_ind + i];
    }

    // Initialize
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < MAX_GROUP_NUM; i++)
        {
            row_group[i] = 0;
            for (int j = 0; j < TILE_WIDTH; j++)
            {
                group_ele_row_idx[i][j] = -1;
            }
        }
    }


    int group_idx = 0;
    BitMaskType and_result; //and_result is used to check if there exists overlap

    BitMaskType expected = row_group[group_idx];
    // or_result is the group mask after adding to the row_group. In this step, the first group is settled.
    BitMaskType or_result = row_group[group_idx] | MatB_bit[entry_ind_bit];
    // Only one row is added to the row_group
    BitMaskType old_value = atomicCAS(&row_group[group_idx], expected, or_result);

    // For rows that haven't been added onto the row_group
    while (expected != old_value) {
        // calculate and_result again to see if there exists overlap
        and_result = row_group[group_idx] & MatB_bit[entry_ind_bit];
        // If there exists overlap, change to next row_group until no overlap exists
        while (and_result != 0) {
            group_idx++;
            if (group_idx >= MAX_GROUP_NUM)
            {
                group_id[entry_ind_bit] = -1;
                int spilled_row_hash_key = atomicAdd(&spilled_row_cnt[bid], 1);
                spilled_row_hash_table_smem[spilled_row_hash_key] = threadIdx.x;
                for (int j = 0; j < TILE_WIDTH; j++)
                {
                    if (d_dense_smem[threadIdx.x][j] != 0)
                    {
                        atomicAdd(&spilled_nnz[bid], 1);
                    }
                }
                break;
            }
            and_result = row_group[group_idx] & MatB_bit[entry_ind_bit];
        }
        if (group_idx >= MAX_GROUP_NUM)
        {
            break;
        }
        expected = row_group[group_idx];
        // Now there is no overlap, try to add onto the new row_group.
        or_result = row_group[group_idx] | MatB_bit[entry_ind_bit];
        old_value = atomicCAS(&row_group[group_idx], expected, or_result);
        group_id[entry_ind_bit] = group_idx;
        // printf("Bid: %d, thread: %d, group_idx: %d\n", bid, threadIdx.x, group_idx);
    }

    for (int i = 0; i < TILE_WIDTH; i++) {
        if ((MatB_bit[entry_ind_bit] >> i & 1) == 0x01) {
            group_ele_row_idx[group_idx][i] = threadIdx.x;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        int spilled_row;

        for (int i = 0; i < spilled_row_cnt[bid]; i++)
        {
            spilled_row = spilled_row_hash_table_smem[i];
            spilled_row_hash_table_reverse_smem[spilled_row] = i;
        }

        // dB_tile_spilled_csrVal[bid] = (float*)malloc(spilled_nnz[bid]);
        // dB_tile_spilled_csrColInd[bid] = (int*)malloc(spilled_nnz[bid]);
        // dB_tile_spilled_csrRowPtr[bid] = (int*)malloc(spilled_row_cnt[bid]+1);

        // printf("bid: %d, nnz: %d\n", bid, spilled_nnz[bid]);

        // load the group information into global memory
        for (int i = 0; i < MAX_GROUP_NUM; i++)
        {
            d_group_mask[MAX_GROUP_NUM * bid + i] = row_group[i];
        }
        for (int i = 0; i < TILE_WIDTH; i++) {
            d_group_ele_row_ind[(MAX_GROUP_NUM * bid + group_idx) * TILE_WIDTH + i] 
                    = group_ele_row_idx[group_idx][i];
            d_group_ele_row_val[(MAX_GROUP_NUM * bid + group_idx) * TILE_WIDTH + i] 
                    = (ValueType)d_dense_smem[group_ele_row_idx[group_idx][i]][i];
        }
        group_id[entry_ind_bit] = group_idx;
    }
    __syncthreads();
    // Load the csr information back to global memory
    spilled_row_hash_table_reverse_gmem[bid * SPLIT_K + threadIdx.x] 
                = spilled_row_hash_table_reverse_smem[threadIdx.x];
    spilled_row_hash_table_gmem[bid * SPLIT_K + threadIdx.x] 
                = spilled_row_hash_table_smem[threadIdx.x];
    // __syncthreads();

}


template <typename BitMaskType,
          typename InitValueType,
          typename ValueType>
__global__ void generate_spilled_csr(BitMaskType *MatB_bit,
                                InitValueType *d_dense,
                                int *group_id,
                                int *spilled_row_cnt,
                                int *spilled_nnz,
                                int *spilled_row_cnt_offset,
                                int *spilled_nnz_offset,
                                ValueType *tile_spilled_csrVal,
                                int *tile_spilled_csrColInd,
                                int *tile_spilled_csrRowPtr,
                                int *spilled_row_hash_table_gmem,
                                int *spilled_row_hash_table_reverse_gmem
                                )
{

    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int row_ind = blockDim.x * blockIdx.y + threadIdx.x;
    int entry_ind = row_ind * gridDim.x * TILE_WIDTH + blockIdx.x * TILE_WIDTH;
    // int entry_ind_bit = row_ind * gridDim.x + blockIdx.x;

    __shared__ InitValueType d_dense_smem[SPLIT_K][TILE_WIDTH];
    __shared__ int spilled_row_hash_table_smem[SPLIT_K];
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        d_dense_smem[threadIdx.x][i] = d_dense[entry_ind + i];
    }
    spilled_row_hash_table_smem[threadIdx.x] = spilled_row_hash_table_gmem[bid * SPLIT_K + threadIdx.x];
    
    if (threadIdx.x == 0)
    {
        int nz_ind_total = 0;
        int row_ind_total = 0;
        int spilled_row;

        // tile_spilled_csrRowPtr[0] = 0;
        for (int i = 0; i < spilled_row_cnt[bid]; i++)
        {
            spilled_row = spilled_row_hash_table_smem[i];
            for (int j = 0; j < TILE_WIDTH; j++)
            {
                if (d_dense_smem[spilled_row][j] != 0)
                {
                    tile_spilled_csrColInd[spilled_nnz_offset[bid] + nz_ind_total] = j;
                    tile_spilled_csrVal[spilled_nnz_offset[bid] + nz_ind_total] = (ValueType)d_dense_smem[spilled_row][j];
                    nz_ind_total++;
                }
            }
            tile_spilled_csrRowPtr[spilled_row_cnt_offset[bid] + row_ind_total] = nz_ind_total;
            row_ind_total++;
        }
    }
}

template <typename InitValueType,
          typename ValueType>
__global__ void csr2tiledcsr(
                int tileA_cnt,
                int64_t dA_nnz,
                int *dA_csr_offset,
                int *dA_csr_column,
                InitValueType *dA_csr_value,
                int *tiled_csr_offset,
                int *tiled_csr_column,
                ValueType *tiled_csr_value,
                int *tile_nnz_acc,
                int *tile_nnz,
                int *tile_row_nnz
                )
{
    // __shared__ int tile_row_nnz[SIZE_M][SIZE_K/SPLIT_K];
    for (int i = 0; i < SIZE_M+1; i++)
    {
        int start_offset = dA_csr_offset[i];
        int end_offset = dA_csr_offset[i+1];
        for (int j = start_offset; j < end_offset; j++)
        {
            int tileA_y = i / TILE_HEIGHT;
            int tileA_x = dA_csr_column[j] / SPLIT_K;
            int tileA_id = tileA_y * (SIZE_K / SPLIT_K) + tileA_x;
            tile_nnz[tileA_id]++;
        }
    }

    int tmp_cnt = 0;
    for (int i = 0; i < tileA_cnt; i++)
    {
        tile_nnz_acc[i] = tmp_cnt;
        tmp_cnt += tile_nnz[i];
    }
    tile_nnz_acc[tileA_cnt] = tmp_cnt;

    int tmp_tile_nnz[(SIZE_M/TILE_HEIGHT)*(SIZE_K/SPLIT_K)];

    for (int i = 0; i < SIZE_M+1; i++)
    {
        int start_offset = dA_csr_offset[i];
        int end_offset = dA_csr_offset[i+1];
        for (int j = start_offset; j < end_offset; j++)
        {
            int tileA_y = i / TILE_HEIGHT;
            int tileA_x = dA_csr_column[j] / SPLIT_K;
            int tileA_id = tileA_y * (SIZE_K / SPLIT_K) + tileA_x;
            tile_row_nnz[i * (SIZE_K / SPLIT_K) + tileA_x]++;

            // int tile_entry_y = i % TILE_HEIGHT;
            int tile_entry_x = dA_csr_column[j] % SPLIT_K;
            // int tile_entry = tile_entry_y * SPLIT_K + tile_entry_x;

            int tile_offset = tile_nnz_acc[tileA_id];
            int entry = tile_offset + tmp_tile_nnz[tileA_id];
            tmp_tile_nnz[tileA_id]++;

            // printf("entry: %d\n", entry);
            tiled_csr_value[entry] = (ValueType)dA_csr_value[j];
            tiled_csr_column[entry] = tile_entry_x;
        }
    }

    for (int i = 0; i < tileA_cnt; i++)
    {
        int tileA_y = i / (SIZE_K/SPLIT_K);
        int tileA_x = i % (SIZE_K/SPLIT_K);
        int tile_nnz_tmp = 0;
        for (int j = 0; j < TILE_HEIGHT+1; j++)
        {
            tiled_csr_offset[(TILE_HEIGHT+1)*i + j] = tile_nnz_tmp; 
            tile_nnz_tmp += tile_row_nnz[(tileA_y*TILE_HEIGHT+j) * (SIZE_K / SPLIT_K) + tileA_x];
        } 
    }
}


template <typename BitMaskType,
          typename ValueType>
__global__ void dense2bitmask(ValueType *MatB_dense, BitMaskType *MatB_bit)
{
    // int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    // int tid = bid * blockDim.x + threadIdx.x;

    int row_ind = blockDim.x * blockIdx.y + threadIdx.x;
    int entry_ind = row_ind * gridDim.x * TILE_WIDTH + blockIdx.x * TILE_WIDTH;
    int entry_ind_bit = row_ind * gridDim.x + blockIdx.x;
    if (TILE_WIDTH == 64) 
    {
        for (int i = 0; i < 64; i++)
        {
            if (MatB_dense[entry_ind + i] != 0)
            {
                atomicOr(&MatB_bit[entry_ind_bit], ((unsigned long long int)1 << i));
            }
        }
    }
    else if (TILE_WIDTH == 32)
    {
        for (int i = 0; i < 32; i++)
        {
            if (MatB_dense[entry_ind + i] != 0)
            {
                atomicOr(&MatB_bit[entry_ind_bit], (1 << i));
            }
        }
    }
}

int dense2CSR(int num_rows, 
                int num_cols, 
                float *&d_dense, 
                float *&d_csr_values, 
                int *&d_csr_offsets, 
                int *&d_csr_columns,
                int64_t &nnzA
                )
{
    int ld = num_cols;
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnzA) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnzA * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnzA * sizeof(float)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    return 0;
}

template <typename ValueType>
int coo2bmp_noTuple_d(const cusp::coo_matrix<int, ValueType, cusp::device_memory>& in,
    cusp::coo_matrix<int, uint64_t, cusp::device_memory>& out,
    thrust::device_vector<ValueType>& elems, thrust::device_vector<uint32_t>& idx) {

    using IndexType = int;
    using ElemIndexType = uint32_t;
    using UnsignedIndexType = uint32_t;
    using LongIndexType = uint64_t;
    using BMPType = uint64_t;
    // using ValueType = float;
    using ValueTypeBMP = uint64_t;
    using COOHostBMP = cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::host_memory>;
    using COODevBMP =  cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::device_memory>;
    using COOHost =    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>;
    using COODev =     cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>;

    auto exec = thrust::cuda::par;

    COODev in_copy(in);

    // sort COO first, it is needed for elem_array. The COO matrix gets sorted by row, and each row by column.
    thrust::sort_by_key(exec, in_copy.column_indices.begin(), in_copy.column_indices.end(), thrust::make_zip_iterator(
            thrust::make_tuple(in_copy.row_indices.begin(), in_copy.values.begin())));
    thrust::stable_sort_by_key(exec, in_copy.row_indices.begin(), in_copy.row_indices.end(), thrust::make_zip_iterator(
            thrust::make_tuple(in_copy.column_indices.begin(), in_copy.values.begin())));

    thrust::device_vector<LongIndexType> tile_indices(in_copy.num_entries); //Absolute index of the tile each element belongs to
    thrust::device_vector<BMPType> position(in_copy.num_entries); //Absolute index of each element inside respective tile (1<<index)

    // Finds 2 things. a) In which tile each element belongs. Tile is returned with absolute indexing. b) What is the
    // position of each element in the respective tile. The position is returned with absolute indexing.
    thrust::for_each(exec,
            thrust::make_zip_iterator(
                    thrust::make_tuple(in_copy.row_indices.begin(), in_copy.column_indices.begin(), tile_indices.begin(),
                            position.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(in_copy.row_indices.end(), in_copy.column_indices.end(), tile_indices.end(), position.end())),
            find_tile_index(in_copy.num_cols));

    // Sort row_indices, col_indices, values and positions in tile by the absolute index of the tile. The sort is stable
    // in order to keep the order of values (elements). The values are expected to come from a COO matrix that has the
    // rows ordered and the columns of each row ordered.
    thrust::stable_sort_by_key(exec, tile_indices.begin(), tile_indices.end(),
            thrust::make_zip_iterator(
                    thrust::make_tuple(in_copy.row_indices.begin(), in_copy.column_indices.begin(), in_copy.values.begin(),
                            position.begin())));

    thrust::device_vector<LongIndexType> tile_indices_unique(in_copy.num_entries); //Unique absolute indices of tiles
    thrust::device_vector<BMPType> bmp(in_copy.num_entries);

    thrust::equal_to<UnsignedIndexType> binary_pred;
    thrust::bit_or<BMPType> binary_op;
    // Elements are reduced based on the index of the tile they belong to. This function returns the unique tile indices and the
    // the result of reduction is the total bmp of all elements that belong to the same tile.
    auto new_end = thrust::reduce_by_key(exec, tile_indices.begin(), tile_indices.end(), position.begin(), tile_indices_unique.begin(),
            bmp.begin(), binary_pred, binary_op);

    UnsignedIndexType num_of_tiles = new_end.first - tile_indices_unique.begin();

    idx.resize(num_of_tiles);

    // transform BMP to population counts
    thrust::transform(exec, bmp.begin(), new_end.second, idx.begin(), bmp_popcount_d());

    // convert population counts to offsets
    thrust::exclusive_scan(exec, idx.begin(), idx.end(), idx.begin(), UnsignedIndexType(0));

    out.num_rows = in_copy.num_rows / BMP_DIM  + ((in_copy.num_rows % BMP_DIM)?1:0) ;
    out.num_cols = in_copy.num_cols / BMP_DIM  + ((in_copy.num_cols % BMP_DIM)?1:0) ;
    out.num_entries = num_of_tiles;
    out.resize(out.num_rows, out.num_cols, out.num_entries);

    // Convert absolute tile indices to relative indexing, to be stored in the COO matrix of the output
    thrust::for_each(exec,
            thrust::make_zip_iterator(
                    thrust::make_tuple(tile_indices_unique.begin(), out.row_indices.begin(), out.column_indices.begin())),
            thrust::make_zip_iterator(
                    thrust::make_tuple(new_end.first, out.row_indices.end(), out.column_indices.end())),
            absolute2relative(out.num_cols));

    thrust::copy(bmp.begin(), new_end.second, out.values.begin());

    elems.resize(in_copy.num_entries);
    thrust::copy(in_copy.values.begin(), in_copy.values.end(), elems.begin());

    return 1;
}


template <typename InputType>
float time_spmmBMP_noTuple(const InputType& A_h, const InputType& B_h)
{
    using IndexType = int;
    using ValueType = signed char;
    using ValueTypeBMP = uint64_t;
    using ElemIndexType = uint32_t;
    using COODevBMP = cusp::coo_matrix<IndexType,ValueTypeBMP,cusp::device_memory>;
    using COOHost   = cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>;
    using COODev    = cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>;

    unsigned int N = REPETITIONS; //repetitions for timing
    const COOHost A_COO_h(A_h), B_COO_h(B_h);

    COODev A_COO_d(A_COO_h);
    COODev B_COO_d(B_COO_h);

    COODevBMP A_BMP_d;
    COODevBMP B_BMP_d;

    thrust::device_vector<ValueType> A_elems_d;
    thrust::device_vector<ValueType> B_elems_d;
    thrust::device_vector<ValueType> C_elems_d; //This is initialized inside the multiply routine

    thrust::device_vector<ElemIndexType> A_idx_d;
    thrust::device_vector<ElemIndexType> B_idx_d;
    thrust::device_vector<ElemIndexType> C_idx_d; //This is initialized inside the multiply routine

    timer t_conv;
    coo2bmp_noTuple_d(A_COO_d, A_BMP_d, A_elems_d, A_idx_d);
    coo2bmp_noTuple_d(B_COO_d, B_BMP_d, B_elems_d, B_idx_d);
    float time_conversion = t_conv.milliseconds_elapsed();
    // printf(" COO to bitmap conversion (for both inputs) time: %lfms\n", time_conversion);

    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        COODevBMP C_BMP_d;
        thrust::device_vector<ValueType> C_elems_d;
        multiplyBmp_noTuple(A_BMP_d, A_elems_d, A_idx_d, B_BMP_d, B_elems_d, B_idx_d, C_BMP_d, C_elems_d, C_idx_d);
    }

    float time_elapsed = t.milliseconds_elapsed() / N;
    return time_elapsed;
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

void initialize_SMatrix(SMatrix *&matrix, int row_size, int col_size, int64_t nnz, 
                  int *csrRowPtr, int *csrColIdx, float *csrVal)
{
    matrix->m = row_size;
    matrix->n = col_size;
    matrix->nnz = nnz;
    matrix->rowpointer = csrRowPtr;
    matrix->columnindex = csrColIdx;
    matrix->value = csrVal;

}

int main() 
{
    // using IndexType = int;
    // using ValueType = float;
    // using CSRHost = cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>;
    // using CSRDev = cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>;

    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
    using InitValueType = float;
    using ValueType = signed char;
    using OutputType = int;

    const int m = SIZE_M;
    const int k = SIZE_K;
	const int n = SIZE_N;

    int tileB_cnt = k * n / SPLIT_K / TILE_WIDTH;
    int tileB_x_cnt = n / TILE_WIDTH;
    int tileB_y_cnt = k / SPLIT_K;

    dim3 grid1(tileB_x_cnt, tileB_y_cnt, 1), block1(SPLIT_K, 1, 1);
    dim3 grid2(SIZE_M/TILE_HEIGHT, SIZE_N/TILE_WIDTH, 1), block2(TILE_HEIGHT, 1, 1);

	InitValueType* hA_dense = (InitValueType*)malloc(sizeof(InitValueType)*m*k);
    InitValueType* hB_dense = (InitValueType*)malloc(sizeof(InitValueType)*k*n);
    // InitValueType* hC_dense = (InitValueType*)malloc(sizeof(InitValueType)*m*n);
    fill_random(hA_dense, m, k, SPARSITY_A);
    fill_random(hB_dense, k, n, SPARSITY_B);
    // fill_random(hC_dense, m, n, SPARSITY);

    // basic ptrs
    InitValueType *dA_dense, *dA_csr_values;
    InitValueType *dB_dense, *dB_csr_values;
    int   *dA_csr_offsets, *dA_csr_columns;
    int   *dB_csr_offsets, *dB_csr_columns;
    int   *dC_csrOffsets, *dC_columns;
    OutputType *dC_values;

    // advanced ptrs
    ValueType *dB_group_ele_val;
    int *dB_group_id, *dB_spilled_row_cnt, *dB_spilled_nnz;
    int *dB_spilled_row_hash_table_gmem, *dB_spilled_row_hash_table_reverse_gmem;
    int *dB_group_ele_ind;
    int *dB_bitmask, *dB_groupmask;

    // basic allocation
    CHECK_CUDA( cudaMalloc((void**) &dA_dense,          m * k * sizeof(InitValueType)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_offsets,   (m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_dense,          k * n * sizeof(InitValueType)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_offsets,   (k + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,    (m + 1) * sizeof(int)) )

    // advanced allocation
    CHECK_CUDA( cudaMalloc((void**) &dB_bitmask,        k * n / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_groupmask,      tileB_cnt * MAX_GROUP_NUM * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_ele_ind,  k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_ele_val,  k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(ValueType)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_id,       k * n / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_cnt,tileB_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_nnz,    tileB_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_hash_table_gmem,
                                    tileB_cnt * SPLIT_K * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_hash_table_reverse_gmem,
                                   tileB_cnt * SPLIT_K * sizeof(int)) )

    
    CHECK_CUDA( cudaMemcpy(dA_dense, hA_dense, m * k * sizeof(InitValueType), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_dense, hB_dense, k * n * sizeof(InitValueType), cudaMemcpyHostToDevice) )
                

    printf("Matrix B dense2bitmask...\n");
    dense2bitmask<<<grid1, block1>>>(dB_dense, dB_bitmask);


    printf("Matrix A dense2CSR...\n");
    int64_t nnzA;

    dense2CSR(m, k, dA_dense, dA_csr_values, dA_csr_offsets, dA_csr_columns, nnzA);

    printf("Transform CSR to tiled CSR\n");
    int tileA_cnt = (SIZE_M/TILE_HEIGHT)*(SIZE_K/SPLIT_K);
    int *dA_tiled_csr_offset, *dA_tiled_csr_column;
    int *dA_tile_nnz_acc, *dA_tile_nnz, *dA_tile_row_nnz;
    ValueType *dA_tiled_csr_value;
    CHECK_CUDA( cudaMalloc((void**) &dA_tile_nnz,         sizeof(int) * tileA_cnt) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tile_nnz_acc,     sizeof(int) * (tileA_cnt+1)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tile_row_nnz,     sizeof(int) * SIZE_M * SIZE_K / SPLIT_K) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tiled_csr_offset, sizeof(int) * tileA_cnt * (TILE_HEIGHT+1)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tiled_csr_column, sizeof(int) * nnzA) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tiled_csr_value,  sizeof(ValueType) * nnzA) )

    csr2tiledcsr<<<1, 1>>>(tileA_cnt, 
                            nnzA, 
                            dA_csr_offsets, 
                            dA_csr_columns, 
                            dA_csr_values,
                            dA_tiled_csr_offset,
                            dA_tiled_csr_column,
                            dA_tiled_csr_value,
                            dA_tile_nnz_acc,
                            dA_tile_nnz,
                            dA_tile_row_nnz
                            );


    int64_t nnzB;
    dense2CSR(k, n, dB_dense, dB_csr_values, dB_csr_offsets, dB_csr_columns, nnzB);

    printf("\nMatrix B generate groups...\n");
    generate_groups<<<grid1, block1>>>(dB_bitmask,                            // input
                                     dB_groupmask,                          // output, for visualization
                                     dB_group_ele_ind,                      // output, not necessary
                                     dB_group_ele_val,                      // output
                                     dB_dense,                              // input
                                     dB_group_id,                           // output
                                     dB_spilled_row_cnt,                    // output
                                     dB_spilled_nnz,
                                    //  dB_tile_spilled_csrVal,                // output
                                    //  dB_tile_spilled_csrColInd,             // output
                                    //  dB_tile_spilled_csrRowPtr,             // output
                                     dB_spilled_row_hash_table_gmem,
                                     dB_spilled_row_hash_table_reverse_gmem // output
                                     );

    int *hB_spilled_nnz = (int*)malloc(tileB_cnt * sizeof(int));
    int *hB_spilled_row_cnt = (int*)malloc(tileB_cnt * sizeof(int));
    cudaMemcpy(hB_spilled_nnz, dB_spilled_nnz, tileB_cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_spilled_row_cnt, dB_spilled_row_cnt, tileB_cnt * sizeof(int), cudaMemcpyDeviceToHost);

    int nnz_cnt = 0;
    int row_cnt = 0;
    int *hB_spilled_nnz_offset = (int*)malloc(tileB_cnt * sizeof(int));
    int *hB_spilled_row_cnt_offset = (int*)malloc(tileB_cnt * sizeof(int));
    ValueType *dB_tile_spilled_csrVal;
    int *dB_tile_spilled_csrColInd, *dB_tile_spilled_csrRowPtr;
    int *dB_spilled_nnz_offset, *dB_spilled_row_cnt_offset;
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_nnz_offset,     tileB_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_cnt_offset,  tileB_cnt * sizeof(int)) )
    for (int i = 0; i < tileB_cnt; i++)
    {
        hB_spilled_nnz_offset[i] = nnz_cnt;
        hB_spilled_row_cnt_offset[i] = row_cnt;
        nnz_cnt += hB_spilled_nnz[i];
        row_cnt += hB_spilled_row_cnt[i];
    }
    cudaMemcpy(dB_spilled_nnz_offset, hB_spilled_nnz_offset, tileB_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_spilled_row_cnt_offset, hB_spilled_row_cnt_offset, tileB_cnt * sizeof(int), cudaMemcpyHostToDevice);

    CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrVal,     nnz_cnt * sizeof(ValueType)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrColInd,  nnz_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrRowPtr,  row_cnt * sizeof(int)) )

    generate_spilled_csr<<<grid1, block1>>>(dB_bitmask,                            // input
                                     dB_dense,                              // input
                                     dB_group_id,                           // output
                                     dB_spilled_row_cnt,                    // output
                                     dB_spilled_nnz,
                                     dB_spilled_row_cnt_offset,
                                     dB_spilled_nnz_offset,
                                     dB_tile_spilled_csrVal,                // output
                                     dB_tile_spilled_csrColInd,             // output
                                     dB_tile_spilled_csrRowPtr,             // output
                                     dB_spilled_row_hash_table_gmem,
                                     dB_spilled_row_hash_table_reverse_gmem // output
                                     );

    std::cout << "Total nnz: " << nnz_cnt << "  Total row_cnt: " << row_cnt << std::endl;
    CHECK_CUDA(cudaDeviceSynchronize())
    ValueType *hB_tile_spilled_csrVal = (ValueType*)malloc(nnz_cnt * sizeof(ValueType));
    int *hB_tile_spilled_csrColInd = (int*)malloc(nnz_cnt * sizeof(int));
    int *hB_tile_spilled_csrRowPtr = (int*)malloc(row_cnt * sizeof(int));

    cudaMemcpy(hB_tile_spilled_csrVal, dB_tile_spilled_csrVal, nnz_cnt * sizeof(ValueType), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_tile_spilled_csrColInd, dB_tile_spilled_csrColInd, nnz_cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_tile_spilled_csrRowPtr, dB_tile_spilled_csrRowPtr, row_cnt * sizeof(int), cudaMemcpyDeviceToHost);

    signed char *d_multiplicand;
    CHECK_CUDA( cudaMalloc((void**) &d_multiplicand,  8 * 16 * sizeof(signed char)) )
    signed char *h_multiplicand = (signed char*)malloc(8 * 16 * sizeof(signed char));
    initialize_multiplicand(h_multiplicand);
    cudaMemcpy(d_multiplicand, h_multiplicand, 8 * 16 * sizeof(signed char), cudaMemcpyHostToDevice);


    OutputType *dC_final_result_gmem;
    CHECK_CUDA( cudaMalloc((void**) &dC_final_result_gmem,  OUTPUT_MAX_GROUP_NUM * SIZE_N * sizeof(OutputType)) )

    int tileC_cnt = SIZE_M * SIZE_N / TILE_HEIGHT / TILE_WIDTH;
    int *dC_spilled_row_cnt, *dC_spilled_nnz, *dC_output_group_idx;
    int *dC_spilled_row_row_idx, *dC_spilled_row_tile_idx;
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_row_cnt,  tileC_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_nnz,  tileC_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_output_group_idx,  SIZE_M * SIZE_N / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_row_row_idx,  MAX_SPILLED_ROW_CNT_C * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_row_tile_idx,  MAX_SPILLED_ROW_CNT_C * sizeof(int)) )

    int *dC_spilled_row_buffersize, *dC_spilled_nnz_buffersize;
    int *dC_spilled_nnz_offset, *dC_spilled_row_cnt_offset;
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_nnz_offset,     (tileC_cnt + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_row_cnt_offset,  (tileC_cnt + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_row_buffersize,  sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_spilled_nnz_buffersize,  sizeof(int)) )

    cudaEvent_t start, end, cusparse_start, cusparse_end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&cusparse_start);
    cudaEventCreate(&cusparse_end);

    // spgemm

    cudaEventRecord(start);
    dim3 grid_2d(SIZE_N/TILE_WIDTH, SIZE_M/TILE_HEIGHT, 1), block_1d(TILE_HEIGHT, 1, 1);
    pre_spgemm<<<grid_2d, block_1d>>>(dB_bitmask, 
                                      dC_spilled_row_cnt, 
                                      dC_spilled_nnz, 
                                      dA_tiled_csr_offset,
                                      dA_tiled_csr_column,  
                                      dA_tile_nnz_acc, 
                                      dC_output_group_idx,
                                      dC_spilled_row_row_idx,
                                      dC_spilled_row_tile_idx,
                                      dC_spilled_row_cnt_offset,
                                      dC_spilled_nnz_offset,
                                      dC_spilled_row_buffersize,
                                      dC_spilled_nnz_buffersize
                                      );

    // printf("pre_spgemm success!\n");
    int *hC_spilled_row_buffersize = (int*)malloc(sizeof(int));
    int *hC_spilled_nnz_buffersize = (int*)malloc(sizeof(int));
    cudaMemcpy(hC_spilled_row_buffersize, dC_spilled_row_buffersize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_spilled_nnz_buffersize, dC_spilled_nnz_buffersize, sizeof(int), cudaMemcpyDeviceToHost);


    int *dC_tile_spilled_csrRowPtr, *dC_tile_spilled_csrColInd;
    float *dC_tile_spilled_csrVal;
    CHECK_CUDA( cudaMalloc((void**) &dC_tile_spilled_csrRowPtr,  *hC_spilled_row_buffersize * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_tile_spilled_csrColInd,  *hC_spilled_nnz_buffersize * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_tile_spilled_csrVal,     *hC_spilled_nnz_buffersize * sizeof(float)) )

    // if (*hC_spilled_row_buffersize != 0)
    // {
    //     spgemm_compute_spilled<<<1, *hC_spilled_row_buffersize>>>(
    //                                     dC_spilled_row_row_idx,
    //                                     dC_spilled_row_tile_idx,
    //                                     dA_tiled_csr_offset,
    //                                     dA_tiled_csr_column,
    //                                     dA_tiled_csr_value,
    //                                     dA_tile_nnz_acc,
    //                                     dB_group_id,
    //                                     dB_bitmask,
    //                                     dB_group_ele_val,
    //                                     dB_spilled_row_hash_table_reverse_gmem,
    //                                     dB_tile_spilled_csrRowPtr,
    //                                     dB_tile_spilled_csrColInd,
    //                                     dB_tile_spilled_csrVal,
    //                                     dB_spilled_row_cnt_offset,
    //                                     dB_spilled_nnz_offset,
    //                                     dC_tile_spilled_csrColInd,
    //                                     dC_tile_spilled_csrVal
    //                                     );
    // }

    // printf("spgemm_compute_spilled success!\n");
    // spgemm_compute_1dthread_tcore<<<grid_2d, block_1d>>>(dB_bitmask, 
    //                                             dA_dense,
    //                                             dB_dense, 
    //                                             dB_group_id, 
    //                                             dB_spilled_row_hash_table_reverse_gmem,
    //                                             dB_group_ele_val,
    //                                             dB_spilled_row_cnt_offset,
    //                                             dB_spilled_nnz_offset,
    //                                             dB_tile_spilled_csrVal,                // output
    //                                             dB_tile_spilled_csrColInd,             // output
    //                                             dB_tile_spilled_csrRowPtr,             // output
    //                                             dA_tiled_csr_offset,
    //                                             dA_tiled_csr_column,
    //                                             dA_tiled_csr_value,
    //                                             dA_tile_nnz_acc,
    //                                             dA_tile_nnz,
    //                                             dC_output_group_idx,
    //                                             dC_final_result_gmem
    //                                             );

    spgemm_compute_1dthread_tcore<<<grid_2d, block_1d>>>(dB_bitmask, 
                                                dB_group_id, 
                                                dB_spilled_row_hash_table_reverse_gmem,
                                                dB_group_ele_val,
                                                dB_spilled_row_cnt_offset,
                                                dB_spilled_nnz_offset,
                                                dB_tile_spilled_csrVal,                // output
                                                dB_tile_spilled_csrColInd,             // output
                                                dB_tile_spilled_csrRowPtr,             // output
                                                dA_tiled_csr_offset,
                                                dA_tiled_csr_column,
                                                dA_tiled_csr_value,
                                                dA_tile_nnz_acc,
                                                dC_output_group_idx,
                                                dC_final_result_gmem,
                                                d_multiplicand
                                                );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    // printf("spgemm_compute success!\n");
    // spgemm_compute<<<grid2, block2>>>(dA_group_indicator_t_gmem, dB_group_ele_val);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    int *hC_spilled_row_cnt = (int*)malloc(tileC_cnt * sizeof(int));
    int *hC_spilled_nnz = (int*)malloc(tileC_cnt * sizeof(int));
    int *hC_output_group_idx = (int*)malloc(SIZE_M * SIZE_N / TILE_WIDTH * sizeof(int));
    cudaMemcpy(hC_spilled_row_cnt, dC_spilled_row_cnt, tileC_cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_spilled_nnz, dC_spilled_nnz, tileC_cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC_output_group_idx, dC_output_group_idx, SIZE_M * SIZE_N / TILE_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    printf("hC_spilled_row_cnt: %d\n\n", *hC_spilled_row_buffersize);

    // for (int i = 0; i < tileC_cnt; i++)
    // {
    //     printf("hC_spilled_row_cnt: %d\n", hC_spilled_row_cnt[i]);
    // }

    // cusparse
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, SIZE_M, SIZE_K, nnzA,
                                      dA_csr_offsets, dA_csr_columns, dA_csr_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, SIZE_K, SIZE_N, nnzB,
                                      dB_csr_offsets, dB_csr_columns, dB_csr_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, SIZE_M, SIZE_N, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseHandle_t     handle = NULL;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    cudaEventRecord(cusparse_start);
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &bufferSize2, dBuffer2) )

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )
    // allocate matrix C
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz1 * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz1 * sizeof(float)) )

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    cudaEventRecord(cusparse_end);
    cudaEventSynchronize(cusparse_end);

    float cusparse_ms;
    cudaEventElapsedTime(&cusparse_ms, cusparse_start, cusparse_end);
    cudaEventDestroy(cusparse_start);
    cudaEventDestroy(cusparse_end);

    cusparseDnMatDescr_t matC_dense;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float                *dC_dense;
    CHECK_CUDA( cudaMalloc((void**) &dC_dense, SIZE_M * SIZE_N * sizeof(float)))
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC_dense, SIZE_M, SIZE_N, SIZE_N, dC_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    CHECK_CUSPARSE( cusparseSparseToDense_bufferSize(
                                        handle, matC, matC_dense,
                                        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseSparseToDense(handle, matC, matC_dense,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC_dense) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    // device result check
    // CHECK_CUDA( cudaMemcpy(hC_dense, dC_dense, SIZE_M * SIZE_N * sizeof(OutputType), cudaMemcpyDeviceToHost) )

    // printMatrixTile(32, 32, SIZE_N, hC_dense, "Mat C ground truth (tile)");

    // tSparse
    dim3 grid_for_convert_A(SIZE_M/32, SIZE_K/32, 1), grid_for_convert_B(SIZE_K/32, SIZE_N/32, 1);
    dim3 block_for_convert(32, 32, 1);
    ValueType *dA_dense_int8, *dB_dense_int8;
    CHECK_CUDA( cudaMalloc((void**) &dA_dense_int8,  SIZE_M * SIZE_K * sizeof(ValueType)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_dense_int8,  SIZE_K * SIZE_N * sizeof(ValueType)) )
    
    format_convert<<<grid_for_convert_A, block_for_convert>>>(dA_dense, dA_dense_int8);
    format_convert<<<grid_for_convert_B, block_for_convert>>>(dB_dense, dB_dense_int8);
    typedef typename cusp::array1d_view< thrust::device_ptr<ValueType> > DeviceArray1dView;
    typedef cusp::array2d_view<DeviceArray1dView, cusp::row_major> DeviceArray2dView;

    // *NOTE* raw pointers must be wrapped with thrust::device_ptr!
    thrust::device_ptr<ValueType> wrapped_device_A(dA_dense_int8);
    // use array1d_view to represent the linear array data
    DeviceArray1dView linear_array_A(wrapped_device_A, wrapped_device_A + SIZE_M*SIZE_K);
    // use array2d_view to wrap the linear array
    DeviceArray2dView A_dense(SIZE_M, SIZE_K, SIZE_K, linear_array_A);

    thrust::device_ptr<ValueType> wrapped_device_B(dB_dense_int8);
    DeviceArray1dView linear_array_B(wrapped_device_B, wrapped_device_B + SIZE_K*SIZE_N);
    DeviceArray2dView B_dense(SIZE_K, SIZE_N, SIZE_N, linear_array_B);

    cusp::coo_matrix<int, ValueType, cusp::host_memory> A_COO_h(SIZE_M, SIZE_K, nnzA);
    cusp::coo_matrix<int, ValueType, cusp::host_memory> B_COO_h(SIZE_K, SIZE_N, nnzB);
    cusp::array2d<ValueType, cusp::host_memory, cusp::row_major> A_dense_h(A_dense);
    cusp::array2d<ValueType, cusp::host_memory, cusp::row_major> B_dense_h(B_dense);

    cusp::convert(A_dense_h, A_COO_h);
    cusp::convert(B_dense_h, B_COO_h);

    float tsparse_ms = time_spmmBMP_noTuple(A_COO_h, B_COO_h);

    // TileSpGEMM
	SMatrix *matrixA = (SMatrix *)malloc(sizeof(SMatrix));
	SMatrix *matrixB = (SMatrix *)malloc(sizeof(SMatrix));
    initialize_SMatrix(matrixA, SIZE_M, SIZE_K, nnzA, dA_csr_offsets, dA_csr_columns, dA_csr_values);
    initialize_SMatrix(matrixB, SIZE_K, SIZE_N, nnzB, dB_csr_offsets, dB_csr_columns, dB_csr_values);

    printf("matrixA-nnz: %d\n", matrixA->nnz);

    printf("bitSparse elapsed time: %fms\n", ms);
    printf("cusparse elapsed time:  %fms\n", cusparse_ms);
    printf("tSparse elpased time:   %fms\n", tsparse_ms);
    printf("\nC_sparsity: %f\n", 1.0 - float(C_nnz1)/SIZE_M/SIZE_N);
    
    // print MatA's information
    if (PRINT_MAT_A_INFO)
    {
        int *hA_csr_offsets = (int*)malloc(sizeof(int) * (m + 1));
        int *hA_csr_columns = (int*)malloc(sizeof(int) * nnzA);
        float *hA_csr_values = (float*)malloc(sizeof(float) * nnzA);
        cudaMemcpy(hA_csr_offsets, dA_csr_offsets, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(hA_csr_columns, dA_csr_columns, sizeof(int) * nnzA, cudaMemcpyDeviceToHost);
        cudaMemcpy(hA_csr_values, dA_csr_values, sizeof(float) * nnzA, cudaMemcpyDeviceToHost);

        std::cout << "nnzA: " << nnzA << std::endl;
        for (int i = 0; i < m+1; i++)
        {
            std::cout << "hA_csr_offsets: " << hA_csr_offsets[i] << std::endl;
        }

        for (int i = 0; i < nnzA; i++)
        {
            std::cout << "hA_csr_columns: " << hA_csr_columns[i] << std::endl;
        }

        for (int i = 0; i < nnzA; i++)
        {
            std::cout << "hA_csr_values: " << hA_csr_values[i] << std::endl;
        }

        int *hA_tiled_csr_offset = (int*)malloc(sizeof(int) * tileA_cnt * (TILE_HEIGHT+1));
        int *hA_tiled_csr_column = (int*)malloc(sizeof(int) * nnzA);
        float *hA_tiled_csr_value = (float*)malloc(sizeof(float) * nnzA);
        int *hA_tile_nnz = (int*)malloc(sizeof(int) * tileA_cnt);
        int *hA_tile_nnz_acc = (int*)malloc(sizeof(int) * (tileA_cnt+1));

        cudaMemcpy(hA_tiled_csr_value, dA_tiled_csr_value, sizeof(float) * nnzA, cudaMemcpyDeviceToHost);
        cudaMemcpy(hA_tiled_csr_column, dA_tiled_csr_column, sizeof(int) * nnzA, cudaMemcpyDeviceToHost);
        cudaMemcpy(hA_tiled_csr_offset, dA_tiled_csr_offset, sizeof(int) * tileA_cnt * (TILE_HEIGHT+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(hA_tile_nnz, dA_tile_nnz, sizeof(int) * tileA_cnt, cudaMemcpyDeviceToHost);
        cudaMemcpy(hA_tile_nnz_acc, dA_tile_nnz_acc, sizeof(int) * (tileA_cnt + 1), cudaMemcpyDeviceToHost);

        printf("nnzA: %ld\n", nnzA);
        for (int i = 0; i < tileA_cnt+1; i++)
        {
            printf("hA_tile_nnz_acc: %d\n", hA_tile_nnz_acc[i]);
        }

        for (int i = 0; i < nnzA; i++)
        {
            printf("hA_tiled_csr_value: %f\n", hA_tiled_csr_value[i]);
        }

        for (int i = 0; i < nnzA; i++)
        {
            printf("hA_tiled_csr_column: %d\n", hA_tiled_csr_column[i]);
        }

        for (int i = 0; i < tileA_cnt * (TILE_HEIGHT+1); i++)
        {
            printf("hA_tiled_csr_offset: %d\n", hA_tiled_csr_offset[i]);
        }
    }

    if (PRINT_MAT_B_INFO)
    {
        int *hB_group_id = (int*)malloc(sizeof(int) * k * n / TILE_WIDTH);
        cudaMemcpy(hB_group_id, dB_group_id, sizeof(int) * k * n / TILE_WIDTH, cudaMemcpyDeviceToHost);

        for (int i = 0; i < SPLIT_K; i++)
        {
            std::cout << "hB_group_id: " << hB_group_id[i] << std::endl;
        }

        // int *hB_tile_spilled_csrRowPtr = (int*)

        int *hB_spilled_row_cnt = (int*)malloc(k * n / SPLIT_K / TILE_WIDTH * sizeof(int));
        cudaMemcpy(hB_spilled_row_cnt, dB_spilled_row_cnt, 
                k * n / SPLIT_K / TILE_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < k * n / SPLIT_K / TILE_WIDTH; i++)
        {
            std::cout << "hB_spilled_row_cnt: " << hB_spilled_row_cnt[i] << std::endl;
        }

        int *hB_spilled_row_hash_table_reverse_gmem = (int*)malloc(k * n / SPLIT_K / TILE_WIDTH * SPLIT_K * sizeof(int));
        cudaMemcpy(hB_spilled_row_hash_table_reverse_gmem, dB_spilled_row_hash_table_reverse_gmem, 
                k * n / SPLIT_K / TILE_WIDTH * SPLIT_K * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < k * n / SPLIT_K / TILE_WIDTH * SPLIT_K; i++)
        {
            std::cout << "hB_spilled_row_hash_table_reverse_gmem-- " << i%SPLIT_K << ": " << hB_spilled_row_hash_table_reverse_gmem[i] << std::endl;
        }

        if (TILE_WIDTH == 64)
        {
            unsigned long long int *hB_bitmask = (unsigned long long int*)malloc(sizeof(unsigned long long int)*k*n/64);
            cudaMemcpy(hB_bitmask, dB_bitmask, k * n / 64 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            printlongintMatrix(k, hB_bitmask, "B_bitmask");
        }
        else if (TILE_WIDTH == 32)
        {
            int *hB_bitmask = (int*)malloc(sizeof(int) * k * n / 32);
            cudaMemcpy(hB_bitmask, dB_bitmask, k * n / 32 * sizeof(int), cudaMemcpyDeviceToHost);
            printintMatrix_32(k, hB_bitmask, "B_bitmask");
        }

        // int *hB_spilled_nnz = (int*)malloc(2 * sizeof(int));
        // cudaMemcpy(hB_spilled_nnz, dB_spilled_nnz, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        // int *hB_tile_spilled_csrColInd = (int*)malloc(hB_spilled_nnz[0] * sizeof(int));
        // cudaMemcpyFromSymbol(hB_tile_spilled_csrColInd, dB_tile_spilled_csrColInd[0], hB_spilled_nnz[0] * sizeof(int), 0, cudaMemcpyDeviceToHost);

        // for (int i = 0; i < hB_spilled_nnz[0]; i++)
        // {
        //     std::cout << "hB_tile_spilled_csrColInd: " << hB_tile_spilled_csrColInd[i] << std::endl;
        // }

    }


    if (TILE_WIDTH == 64)
    {
        unsigned long long int *hB_groupmask = 
        (unsigned long long int*)malloc(k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(unsigned long long int));
        cudaMemcpy(hB_groupmask, dB_groupmask, k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    }
    else if (TILE_WIDTH == 32)
    {
        int *hB_groupmask = (int*)malloc(k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int));
        cudaMemcpy(hB_groupmask, dB_groupmask, k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int), cudaMemcpyDeviceToHost);
        printintMatrix_32(16, hB_groupmask, "B_groupmask");

        std::cout << "A random number: " << rand() % 100 << std::endl;
        int *hB_group_ele_ind = (int*)malloc(k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(int));
        cudaMemcpy(hB_group_ele_ind, dB_group_ele_ind, k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(int), cudaMemcpyDeviceToHost);

    }
    
    // size_t *size;
    // cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
    // std::cout << "HeapSize: " << *size << std::endl;

    // free(dB)

    // std::cout << "Input matrix A has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n";
    // std::cout << "             B has shape (" << B.num_rows << "," << B.num_cols << ") and " << B.num_entries << " entries" << "\n\n";

}