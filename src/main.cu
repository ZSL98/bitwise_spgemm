#include <cuda_runtime_api.h>
#include <cusparse.h>

#include "common.h"
#include "cusp/csr_matrix.h"

// __global__ void test_kernel(int *d_group_mask)
// {
//     d_group_mask[0] = 1024;
//     d_group_mask[1] = 1;
//     d_group_mask[2] = 1;
//     d_group_mask[3] = 1;
// }

__device__ float* dB_tile_spilled_csrVal[2];
__device__ int* dB_tile_spilled_csrColInd[2];
__device__ int* dB_tile_spilled_csrRowPtr[2];


template <typename int32_or_64>
__global__ void generate_groups(int32_or_64 *MatB_bit,
                                int32_or_64 *d_group_mask,
                                int *d_group_ele_row_ind,
                                float *d_group_ele_row_val,
                                float *d_dense,
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
    int tid = bid * blockDim.x + threadIdx.x;
    int row_ind = blockDim.x * blockIdx.y + threadIdx.x;
    int entry_ind = row_ind * gridDim.x * TILE_WIDTH + blockIdx.x * TILE_WIDTH;
    int entry_ind_bit = row_ind * gridDim.x + blockIdx.x;

    __shared__ int row_group[MAX_GROUP_NUM];
    __shared__ int group_ele_row_idx[MAX_GROUP_NUM][TILE_WIDTH];
    __shared__ float d_dense_smem[SPLIT_K][TILE_WIDTH];
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
    int32_or_64 and_result; //and_result is used to check if there exists overlap

    int32_or_64 expected = row_group[group_idx];
    // or_result is the group mask after adding to the row_group. In this step, the first group is settled.
    int32_or_64 or_result = row_group[group_idx] | MatB_bit[entry_ind_bit];
    // Only one row is added to the row_group
    int32_or_64 old_value = atomicCAS(&row_group[group_idx], expected, or_result);

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
                    if (d_dense_smem[threadIdx.x][j] != 0.0f)
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

        // for (int i = 0; i < spilled_nnz[bid]; i++)
        // {
        //     dB_tile_spilled_csrColInd[bid][i] = 0;
        //     dB_tile_spilled_csrVal[bid][i] = 0;
        // }

        int nz_ind;
        int spilled_row;

        for (int i = 0; i < spilled_row_cnt[bid]; i++)
        {
            spilled_row = spilled_row_hash_table_smem[i];
            spilled_row_hash_table_reverse_smem[spilled_row] = i;
        }

        dB_tile_spilled_csrVal[bid] = (float*)malloc(spilled_nnz[bid]);
        dB_tile_spilled_csrColInd[bid] = (int*)malloc(spilled_nnz[bid]);
        dB_tile_spilled_csrRowPtr[bid] = (int*)malloc(spilled_row_cnt[bid]+1);

        // nz_ind = 0;
        // for (int i = 0; i < spilled_row_cnt[bid]; i++)
        // {
        //     spilled_row = spilled_row_hash_table_smem[i];
        //     spilled_row_hash_table_reverse_smem[spilled_row] = i;
        //     for (int j = 0; j < TILE_WIDTH; j++)
        //     {
        //         if (d_dense_smem[spilled_row][j] >= 1.0f)
        //         {
        //             // dB_tile_spilled_csrColInd[bid][nz_ind] = j;
        //             dB_tile_spilled_csrVal[bid][nz_ind] = d_dense_smem[spilled_row][j];
        //             // printf("bid: %d, d_dense_smem: %f\n", bid, d_dense_smem[spilled_row][j]);
        //             // dB_tile_spilled_csrVal[bid][nz_ind] = j;
        //             nz_ind++;
        //         }
        //     }
        // }

        // dB_tile_spilled_csrRowPtr[bid] = (int*)malloc(spilled_row_cnt[bid]+1);
        // dB_tile_spilled_csrRowPtr[bid][0] = 0;
        // nz_ind = 0;
        // for (int i = 0; i < spilled_row_cnt[bid]; i++)
        // {
        //     spilled_row = spilled_row_hash_table_smem[i];
        //     for (int j = 0; j < TILE_WIDTH; j++)
        //     {
        //         if (d_dense_smem[spilled_row][j] >= 1.0f)
        //         {
        //             nz_ind++;
        //         }
        //     }
        //     dB_tile_spilled_csrRowPtr[bid][i+1] = nz_ind;
        // }      

        // dB_tile_spilled_csrColInd[bid] = (int*)malloc(spilled_nnz[bid]);
        // nz_ind = 0;
        // for (int i = 0; i < spilled_row_cnt[bid]; i++)
        // {
        //     spilled_row = spilled_row_hash_table_smem[i];
        //     for (int j = 0; j < TILE_WIDTH; j++)
        //     {
        //         if (d_dense_smem[spilled_row][j] >= 1.0f)
        //         {
        //             dB_tile_spilled_csrColInd[bid][nz_ind] = j;
        //             nz_ind++;
        //         }
        //     }
        // }

        printf("bid: %d, nnz: %d\n", bid, spilled_nnz[bid]);

        // for (int i = 0; i < spilled_nnz[bid]; i++)
        // {
        //     printf("bid: %d, i: %d, csrColInd2: %f \n", bid, i, dB_tile_spilled_csrVal[bid][i]);
        // }

        // load the group information into global memory
        for (int i = 0; i < MAX_GROUP_NUM; i++)
        {
            d_group_mask[MAX_GROUP_NUM * bid + i] = row_group[i];
        }
        for (int i = 0; i < TILE_WIDTH; i++) {
            d_group_ele_row_ind[(MAX_GROUP_NUM * bid + group_idx) * TILE_WIDTH + i] 
                    = group_ele_row_idx[group_idx][i];
            d_group_ele_row_val[(MAX_GROUP_NUM * bid + group_idx) * TILE_WIDTH + i] 
                    = d_dense_smem[group_ele_row_idx[group_idx][i]][i];
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


template <typename int32_or_64>
__global__ void generate_groups_step2(int32_or_64 *MatB_bit,
                                int32_or_64 *d_group_mask,
                                int *d_group_ele_row_ind,
                                float *d_group_ele_row_val,
                                float *d_dense,
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
    int tid = bid * blockDim.x + threadIdx.x;
    int row_ind = blockDim.x * blockIdx.y + threadIdx.x;
    int entry_ind = row_ind * gridDim.x * TILE_WIDTH + blockIdx.x * TILE_WIDTH;
    int entry_ind_bit = row_ind * gridDim.x + blockIdx.x;

    __shared__ float d_dense_smem[SPLIT_K][TILE_WIDTH];
    __shared__ int spilled_row_hash_table_smem[SPLIT_K];
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        d_dense_smem[threadIdx.x][i] = d_dense[entry_ind + i];
    }
    spilled_row_hash_table_smem[threadIdx.x] = spilled_row_hash_table_gmem[bid * SPLIT_K + threadIdx.x];
    
    if (threadIdx.x == 0)
    {
        int nz_ind = 0;
        int spilled_row;

        dB_tile_spilled_csrRowPtr[bid][0] = 0;
        for (int i = 0; i < spilled_row_cnt[bid]; i++)
        {
            spilled_row = spilled_row_hash_table_smem[i];
            for (int j = 0; j < TILE_WIDTH; j++)
            {
                if (d_dense_smem[spilled_row][j] != 0.0f)
                {
                    dB_tile_spilled_csrColInd[bid][nz_ind] = j;
                    dB_tile_spilled_csrVal[bid][nz_ind] = d_dense_smem[spilled_row][j];
                    nz_ind++;
                }
            }
            dB_tile_spilled_csrRowPtr[bid][i+1] = nz_ind;
        }     
    }
}

// __device__ void ld_groups_to_regs(int *d_group_ele_row_idx, 
//                                   unsigned long long int *d_group_mask,
//                                   float *dB_dense
//                                     )
// {
//     // int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // column per thread
//     // Load rows to groups and // add up groups into registers
//     for (int i = 0; i < MAX_GROUP_NUM; i++) {
//         // Need modification, should read from the CSR or other condensed formats.
//         group[i][tid] = dB_dense[d_group_ele_row_idx[i][tid]][tid];
//     }
    
//     // MAX_GROUP_NUM = 4
//     group[4][tid] = group[0][tid] + group[1][tid];
//     group[5][tid] = group[0][tid] + group[2][tid];
//     group[6][tid] = group[0][tid] + group[3][tid];
//     group[7][tid] = group[1][tid] + group[2][tid];
//     group[8][tid] = group[1][tid] + group[3][tid];
//     group[9][tid] = group[2][tid] + group[3][tid];
//     group[10][tid] = group[4][tid] + group[2][tid];
//     group[11][tid] = group[4][tid] + group[3][tid];
//     group[12][tid] = group[5][tid] + group[3][tid];
//     group[13][tid] = group[7][tid] + group[3][tid];
//     group[14][tid] = group[10][tid] + group[3][tid];
//     group[15][tid] = group[4][tid] + group[9][tid];

// }

template <typename int32_or_64>
__global__ void bit_wise_spgemm(int split_k,
                                float *d_csr_values, 
                                int *d_csr_offsets, 
                                int *d_csr_columns,
                                float *d_group_ele_row_val,
                                int32_or_64 *MatB_bit,           // MatrixB's bit mask
                                int *group_id_gmem,                         // MatrixB's group ID
                                int *spilled_row_cnt,
                                int *spilled_nnz,
                                // float **tile_spilled_csrVal,
                                // int **tile_spilled_csrColInd,
                                // int **tile_spilled_csrRowPtr,
                                int *spilled_row_hash_table_reverse_gmem
                                )
{
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = bid * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < spilled_nnz[bid]; i++)
        {
            printf("bid: %d, i: %d, csrColInd2: %d \n", bid, i, dB_tile_spilled_csrColInd[bid][i]);
        }
    }

    int assigned_row_ind = blockIdx.y * blockDim.x + threadIdx.x;
    // int assigned_col_ind = blockIdx.x * split_k;
    // int assigned_bit_pos = threadIdx.x % BIT_WIDTH;
    int entry_ind_bit = assigned_row_ind * gridDim.x + blockIdx.x;

    // printf("bid: %d, tid: %d, d_csr_values: %f \n", bid, tid, d_csr_values[tid]);

    int row_ind_in_tile, row_group_id, register_idx, col_ind;
    __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];

    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        for (int i = 0; i < blockDim.x; i++) 
        {
            for (int b = 0; b < BIT_WIDTH; b++)
            {
                for (int j = 0; j < MAX_GROUP_NUM; j++) 
                {
                    group_indicator[i][b][j] = 0;
                }
            }
        }

        int tileB_id = k * gridDim.x + blockIdx.x;
        // Load group_id to shared memory
        // Load spilled_row_hash_table_reverse to shared memory
        for (int i = 0; i < SPLIT_K/blockDim.x; i++)
        {
            int row_ind = k * SPLIT_K + i * blockDim.x + threadIdx.x;
            int entry = row_ind * gridDim.x + blockIdx.x;
            // Note that the layout of group_id is row major while the layout of spilled_row_hash_table is block major
            group_id_smem[i * blockDim.x + threadIdx.x] = group_id_gmem[entry];
            spilled_row_hash_table_reverse_smem[i * blockDim.x + threadIdx.x] 
                = spilled_row_hash_table_reverse_gmem[i * blockDim.x + threadIdx.x + tileB_id * SPLIT_K];
        }

        // Transform an specific area of the CSR of MatrixA to a tiled form.
        // The transformation process is inherited in this kernel
        for (int i = d_csr_offsets[assigned_row_ind]; i < d_csr_offsets[assigned_row_ind+1]; i++)
        {
            if (d_csr_columns[i] > SPLIT_K * k && d_csr_columns[i] < SPLIT_K * (k+1))
            {
                row_ind_in_tile = d_csr_columns[i] - SPLIT_K * k;
                // printf("bid: %d, threadIdx.x: %d, row_ind_in_tile: %d \n", bid, threadIdx.x, row_ind_in_tile);
                for (int b = 0; b < BIT_WIDTH; b++)
                {
                    int tmp = __float_as_int(d_csr_values[i]);
                    int mv_bit = b+16;
                    if((tmp >> mv_bit) & 1 == 0x01)
                    {
                        row_group_id = group_id_smem[row_ind_in_tile];
                        // printf("bid: %d, threadIdx.x: %d, b: %d, row_group_id: %d \n", bid, threadIdx.x, b, row_group_id);
                        if (row_group_id != -1)
                        {
                            int entry = ((TILE_HEIGHT * k) + row_ind_in_tile) * gridDim.x + blockIdx.x;
                            atomicOr(&group_indicator[threadIdx.x][b][row_group_id], MatB_bit[entry]);
                            // printf("bid: %d, entry: %d, b: %d, Matbit: %d \n", bid, entry, b, group_indicator[threadIdx.x][b][row_group_id]);
                        }
                        else 
                        {
                            // Current row_ind_in_tile in MatrixB is the spilled row
                            // Perform the extra computation
                            // printf("bid: %d, threadIdx.x: %d, row_ind_in_tile: %d \n", bid, threadIdx.x, row_ind_in_tile);
                            int row_in_csr = spilled_row_hash_table_reverse_smem[row_ind_in_tile];
                            // printf("tileB_id: %d, row_ind_in_tile: %d, row_in_csr: %d, Ptr: %d, Ptr2: %d \n", 
                            // tileB_id, row_ind_in_tile, row_in_csr, dB_tile_spilled_csrRowPtr[tileB_id][row_in_csr], dB_tile_spilled_csrRowPtr[tileB_id][row_in_csr+1]);
                            for (int j = dB_tile_spilled_csrRowPtr[tileB_id][row_in_csr]; 
                                    j < dB_tile_spilled_csrRowPtr[tileB_id][row_in_csr+1]; j++)
                            {
                                col_ind = dB_tile_spilled_csrColInd[tileB_id][j];
                                // printf("col_ind: %d \n", col_ind);
                                // result[threadIdx.x][b][col_ind] += dB_tile_spilled_csrVal[tileB_id][col_ind];
                            }
                        }
                    }
                }
            }
        }

        // printf("bid: %d, Matbit: %d \n", bid, group_indicator[threadIdx.x][5][0]);

        // for (int i = 0; i < 8; i++)
        // {
        //     printf("bid: %d, group_indicator: %d \n", bid, group_indicator[0][i][threadIdx.x]);
        // }

        float group[MAX_GROUP_NUM];
        // Load groups to registers, one column per thread
#pragma unroll
        for (int i = 0; i < MAX_GROUP_NUM; i++)
        {
            group[1 << i] = d_group_ele_row_val[(MAX_GROUP_NUM * tileB_id + i) * TILE_WIDTH + threadIdx.x];
        }
        // Calculate the combinations of groups and store the results in registers if possible
        // ...


        // One column per thread to read values from registers
        // To achieve so, the tile_width and the tile_height should be the same
        for (int i = 0; i < blockDim.x; i++) 
        {
            for (int b = 0; b < BIT_WIDTH; b++)
            {
                for (int j = 0; j < MAX_GROUP_NUM; j++) 
                {
                    // register_idx = atomicAnd(&register_idx, (group_indicator[i][b][j] << threadIdx.x)); // this is one bit
                    
                    // if (group_indicator[i][b][j] == 1)
                    // {
                    //     register_idx = group_indicator[i][b][j] << threadIdx.x;
                    //     result[i][b][threadIdx.x] += group[register_idx];
                    // }
                }
                // result[i][b][tid] = group[register_idx];
            }

        }
    }

}


template <typename int32_or_64>
__global__ void dense2bitmask(float *MatB_dense, int32_or_64 *MatB_bit)
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
            if (MatB_dense[entry_ind + i] != 0.0f)
            {
                atomicOr(&MatB_bit[entry_ind_bit], ((unsigned long long int)1 << i));
            }
        }
    }
    else if (TILE_WIDTH == 32)
    {
        for (int i = 0; i < 32; i++)
        {
            if (MatB_dense[entry_ind + i] != 0.0f)
            {
                atomicOr(&MatB_bit[entry_ind_bit], (1 << i));
            }
        }
    }
}

int dense2CSR(int num_rows, 
                int num_cols, 
                float *d_dense, 
                float *d_csr_values, 
                int *d_csr_offsets, 
                int *d_csr_columns,
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


int main() 
{
    // using IndexType = int;
    // using ValueType = float;
    // using CSRHost = cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>;
    // using CSRDev = cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
    dim3 grid1(2, 1, 1), block1(SPLIT_K, 1, 1);
    // dim3 grid1(SIZE_K/SPLIT_K, SIZE_N/TILE_WIDTH, 1), block1(SPLIT_K, 1, 1);
    dim3 grid2(SIZE_K/TILE_HEIGHT, SIZE_N/TILE_WIDTH, 1), block2(TILE_HEIGHT, 1, 1);

    const int m = SIZE_M;
    const int k = SIZE_K;
	const int n = SIZE_N;

    int sparsity = 90;
	float* hA_dense = (float*)malloc(sizeof(float)*m*k);
    float* hB_dense = (float*)malloc(sizeof(float)*k*n);
    float* hC_dense = (float*)malloc(sizeof(float)*m*n);
    fill_random(hA_dense, m, k, sparsity);
    fill_random(hB_dense, k, n, sparsity);
    fill_random(hC_dense, m, n, sparsity);

    float *dA_dense, *dA_csr_values, *dB_dense, *dB_group_ele_val;
    int   *dA_csr_offsets, *dA_csr_columns, *dB_group_id, *dB_spilled_row_cnt, *dB_spilled_nnz, *dB_spilled_row_hash_table_gmem, *dB_spilled_row_hash_table_reverse_gmem;

    int *dB_group_ele_ind;
    // float **dB_tile_spilled_csrVal;
    // int **dB_tile_spilled_csrColInd, **dB_tile_spilled_csrRowPtr;

    // unsigned long long int *dB_bitmask, *dB_groupmask;
    // CHECK_CUDA( cudaMalloc((void**) &dB_bitmask,        k * n / TILE_WIDTH * sizeof(unsigned long long int)) )
    // CHECK_CUDA( cudaMalloc((void**) &dB_groupmask,      k * n / TILE_HEIGHT / TILE_WIDTH * MAX_GROUP_NUM * sizeof(unsigned long long int)) )

    int *dB_bitmask, *dB_groupmask;
    CHECK_CUDA( cudaMalloc((void**) &dB_bitmask,        k * n / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_groupmask,      k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int)) )
    

    CHECK_CUDA( cudaMalloc((void**) &dA_dense,          m * k * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_offsets,   (m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_dense,          k * n * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_ele_ind,  k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_ele_val,  k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_id,       k * n / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_cnt,k * n / SPLIT_K / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_nnz,    k * n / SPLIT_K / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_hash_table_gmem,
                                    k * n / SPLIT_K / TILE_WIDTH * SPLIT_K * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_hash_table_reverse_gmem,
                                    k * n / SPLIT_K / TILE_WIDTH * SPLIT_K * sizeof(int)) )

    // CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrVal,     k * n / SPLIT_K / TILE_WIDTH * sizeof(float*)) )
    // CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrColInd,  k * n / SPLIT_K / TILE_WIDTH * sizeof(int*)) )
    // CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrRowPtr,  k * n / SPLIT_K / TILE_WIDTH * sizeof(int*)) )
    
    CHECK_CUDA( cudaMemcpy(dA_dense, hA_dense, m * k * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_dense, hB_dense, k * n * sizeof(float),
                           cudaMemcpyHostToDevice) )
                

    printf("Matrix B dense2bitmask...\n");
    dense2bitmask<<<grid1, block1>>>(dB_dense, dB_bitmask);

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

    printf("Matrix A dense2CSR...\n");
    int64_t nnzA;

    // dense2CSR(m, k, dA_dense, dA_csr_values, dA_csr_offsets, dA_csr_columns, nnzA);

    int num_cols = k;
    int num_rows = m;
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
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, num_rows, num_cols, ld, dA_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      dA_csr_offsets, NULL, NULL,
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
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_columns, nnzA * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_values,  nnzA * sizeof(float)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, dA_csr_offsets, dA_csr_columns,
                                           dA_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )


    // test_kernel<<<grid1, block1>>>(dB_groupmask);
    // int *hB_groupmask = (int*)malloc(k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int));
    // cudaMemcpy(hB_groupmask, dB_groupmask, k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int), cudaMemcpyDeviceToHost);
    // printintMatrix_32(16, hB_groupmask, "B_groupmask");


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

    generate_groups_step2<<<grid1, block1>>>(dB_bitmask,                            // input
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

    // int *hB_groupmask = (int*)malloc(k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int));
    // cudaMemcpy(hB_groupmask, dB_groupmask, k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int), cudaMemcpyDeviceToHost);
    // printintMatrix_32(16, hB_groupmask, "B_groupmask");

    // spgemm
    bit_wise_spgemm<<<grid2, block2>>>(SPLIT_K, 
                                    dA_csr_values, 
                                    dA_csr_offsets, 
                                    dA_csr_columns, 
                                    dB_group_ele_val, 
                                    dB_bitmask, 
                                    dB_group_id, 
                                    dB_spilled_row_cnt,
                                    dB_spilled_nnz,
                                    // dB_tile_spilled_csrVal, 
                                    // dB_tile_spilled_csrColInd, 
                                    // dB_tile_spilled_csrRowPtr, 
                                    dB_spilled_row_hash_table_reverse_gmem
                                    );


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
        for (int i = 0; i < 32; i++)
        {
            std::cout << std::left << std::setw(4) << hB_group_ele_ind[i] << std::endl;
        }
    }
    
    size_t *size;
    cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
    std::cout << "HeapSize: " << *size << std::endl;

    // free(dB)

    // std::cout << "Input matrix A has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n";
    // std::cout << "             B has shape (" << B.num_rows << "," << B.num_cols << ") and " << B.num_entries << " entries" << "\n\n";

}