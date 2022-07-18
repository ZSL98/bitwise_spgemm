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

// __device__ float* dB_tile_spilled_csrVal[2];
// __device__ int* dB_tile_spilled_csrColInd[2];
// __device__ int* dB_tile_spilled_csrRowPtr[2];


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
        int spilled_row;

        for (int i = 0; i < spilled_row_cnt[bid]; i++)
        {
            spilled_row = spilled_row_hash_table_smem[i];
            spilled_row_hash_table_reverse_smem[spilled_row] = i;
        }

        // dB_tile_spilled_csrVal[bid] = (float*)malloc(spilled_nnz[bid]);
        // dB_tile_spilled_csrColInd[bid] = (int*)malloc(spilled_nnz[bid]);
        // dB_tile_spilled_csrRowPtr[bid] = (int*)malloc(spilled_row_cnt[bid]+1);

        printf("bid: %d, nnz: %d\n", bid, spilled_nnz[bid]);

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
__global__ void generate_spilled_csr(int32_or_64 *MatB_bit,
                                int32_or_64 *d_group_mask,
                                int *d_group_ele_row_ind,
                                float *d_group_ele_row_val,
                                float *d_dense,
                                int *group_id,
                                int *spilled_row_cnt,
                                int *spilled_nnz,
                                int *spilled_row_cnt_offset,
                                int *spilled_nnz_offset,
                                float *tile_spilled_csrVal,
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

    __shared__ float d_dense_smem[SPLIT_K][TILE_WIDTH];
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
                if (d_dense_smem[spilled_row][j] != 0.0f)
                {
                    tile_spilled_csrColInd[spilled_nnz_offset[bid] + nz_ind_total] = j;
                    tile_spilled_csrVal[spilled_nnz_offset[bid] + nz_ind_total] = d_dense_smem[spilled_row][j];
                    nz_ind_total++;
                }
            }
            tile_spilled_csrRowPtr[spilled_row_cnt_offset[bid] + row_ind_total] = nz_ind_total;
            row_ind_total++;
        }
    }

    // if (threadIdx.x == 0)
    // {
    //     for (int i = 0; i < spilled_nnz[bid]; i++)
    //     {
    //         printf("bid: %d, tile_spilled_csrColInd: %f\n", bid, tile_spilled_csrVal[spilled_nnz_offset[bid] + nz_ind_total]);
    //     }
    // }
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
__global__ void bit_wise_spgemm_3d(int split_k,
                                float *d_csr_values, 
                                int *d_csr_offsets, 
                                int *d_csr_columns,
                                float *d_group_ele_row_val,
                                int32_or_64 *MatB_bit,           // MatrixB's bit mask
                                int *group_id_gmem,                         // MatrixB's group ID
                                int *spilled_row_cnt,
                                int *spilled_nnz,
                                int *spilled_row_cnt_offset,
                                int *spilled_nnz_offset,
                                float *tile_spilled_csrVal,
                                int *tile_spilled_csrColInd,
                                int *tile_spilled_csrRowPtr,
                                int *spilled_row_hash_table_reverse_gmem,
                                float *final_result_gmem
                                )
{

    int assigned_row_ind = blockIdx.y * blockDim.x + threadIdx.x;

    __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];

    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        if (threadIdx.z < MAX_GROUP_NUM)
        {
            group_indicator[threadIdx.x][threadIdx.y][threadIdx.z] = 0;
        }

        
        int tileB_id = k * gridDim.x + blockIdx.x;
        // Load group_id to shared memory
        // Load spilled_row_hash_table_reverse to shared memory
        if ((threadIdx.y == 0) && (threadIdx.z == 0))
        {
            for (int i = 0; i < SPLIT_K/blockDim.x; i++)
            {
                int row_ind = k * SPLIT_K + i * blockDim.x + threadIdx.x;
                int entry = row_ind * gridDim.x + blockIdx.x;
                // Note that the layout of group_id is row major while the layout of spilled_row_hash_table is block major
                group_id_smem[i * blockDim.x + threadIdx.x] = group_id_gmem[entry];
                spilled_row_hash_table_reverse_smem[i * blockDim.x + threadIdx.x] 
                    = spilled_row_hash_table_reverse_gmem[i * blockDim.x + threadIdx.x + tileB_id * SPLIT_K];
            }
        }


    }
}


__global__ void generate_tiled_csr(
                float *d_csr_values,
                int *d_csr_columns,
                int *d_csr_offsets
            )
{
    int row_ind = blockIdx.y * blockDim.x + threadIdx.x;
    int start_offset = d_csr_offsets[row_ind];

    for (int i = 0; i < SPLIT_K; i++)
    {

    }
}


template <typename int32_or_64>
__global__ void generate_group_indicator(
                int32_or_64 *MatB_bit,
                float *dA_dense_gmem,
                int *group_id_gmem,
                int *spilled_row_hash_table_reverse_gmem,
                float *d_group_ele_row_val,
                int *spilled_row_cnt_offset,
                int *spilled_nnz_offset,
                float *tile_spilled_csrVal,
                int *tile_spilled_csrColInd,
                int *tile_spilled_csrRowPtr,
                float *final_result_gmem
            )
{
    // __shared__ float dA_dense_smem[TILE_HEIGHT][SPLIT_K];
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];
    __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    // __shared__ int group_indicator_t[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ float group[MAX_GROUP_NUM][TILE_WIDTH];
    __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ int32_or_64 MatB_bit_smem[SPLIT_K];
    
    int row_group_id;
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    
    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        int tileA_id = gridDim.x * blockIdx.y + k;
        int tileB_id = k * gridDim.x + blockIdx.x;

        // Load MatB's group data into shared memory
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int z = 0; z < TILE_WIDTH; z++)
            {
                int entry_B = ((SPLIT_K * k) + z) * gridDim.x + blockIdx.x;
                MatB_bit_smem[z] = MatB_bit[entry_B];
                for (int i = 0; i < MAX_GROUP_NUM; i++)
                {
                    group[i][z] = d_group_ele_row_val[(MAX_GROUP_NUM * tileB_id + i) * TILE_WIDTH + z];
                }
            }
        }

        // Load MatB's group information into shared memory
        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        int rowB_ind = k * SPLIT_K + tid;
        int entry = rowB_ind * gridDim.x + blockIdx.x;
        group_id_smem[tid] = group_id_gmem[entry];
        spilled_row_hash_table_reverse_smem[tid] 
            = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        int rowA_ind = blockIdx.y * blockDim.x + threadIdx.x;
        for (int z = 0; z < SPLIT_K; z++)
        {
            int entry_A = rowA_ind * SIZE_K + k * SPLIT_K + z;
            if (dA_dense_gmem[entry_A] == 0.0f)
            {
                continue;
            }
            // printf("entry_A: %d\n", entry_A);
            int tmp = (__float_as_int(dA_dense_gmem[entry_A]) >> threadIdx.y) & 1 == 0x01;
            // if ((__float_as_int(dA_dense_gmem[entry]) >> threadIdx.y) & 1 == 0x01)
            if ((threadIdx.y % 2 + z % 2) % 2 == 0)
            {
                row_group_id = group_id_smem[z];
                if (row_group_id != -1)
                {
                    atomicOr(&group_indicator[threadIdx.x][threadIdx.y][row_group_id], MatB_bit_smem[z]);
                }
                else 
                {
                    // Current row_ind_in_tile in MatrixB is the spilled row
                    // Perform the extra computation
                    int row_in_csr = spilled_row_hash_table_reverse_smem[z];
                    int start_offset;
                    if (row_in_csr == 0)
                    {
                        start_offset = 0;
                    }
                    else 
                    {
                        start_offset = tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                    }
                    for (int j = start_offset; j < tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                    {
                        int col_ind = tile_spilled_csrColInd[spilled_nnz_offset[tileB_id] + j];
                        result[threadIdx.x][threadIdx.y][col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
                    }
                }
            }
        }

        for (int z = 0; z < TILE_WIDTH; z++)
        {
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                if (((group_indicator[threadIdx.x][threadIdx.y][i] >> z) & 0x01) == 1)
                {
                    result[threadIdx.x][threadIdx.y][z] += group[i][z];
                    // result[threadIdx.x][threadIdx.y][z] += i;
                }
            }
        }
    }

    // compute with cuda core
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        int ind = (blockIdx.y * blockDim.x + threadIdx.x) * SIZE_N + blockIdx.x * TILE_WIDTH + i;
        final_result_gmem[ind] += result[threadIdx.x][threadIdx.y][i] * float(threadIdx.y);
    }

}


__global__ void dense2tiledcsr_step1(
                float *dA_dense_gmem,
                int *tiled_csr_offset,
                int *tiled_csr_column,
                float *tiled_csr_value,
                int *tiled_csr_nnz
                )
{
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = 0; i < TILE_HEIGHT; i++)
    {
        for (int j = 0; j < SPLIT_K; j++)
        {
            int row_ind = blockIdx.y + i;
            int entry = row_ind * SIZE_K + blockIdx.x * SPLIT_K + j;
            if (dA_dense_gmem[entry] != 0.0f)
            {
                tiled_csr_nnz[bid]++;
            }
        }
    }
}

// __global__ void dense2tiledcsr_step2(

// )


__global__ void csr2tiledcsr(
                int tileA_cnt,
                int64_t dA_nnz,
                int *dA_csr_offset,
                int *dA_csr_column,
                float *dA_csr_value,
                int *tiled_csr_offset,
                int *tiled_csr_column,
                float *tiled_csr_value,
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
            tiled_csr_value[entry] = dA_csr_value[j];
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


template <typename int32_or_64>
__global__ void generate_group_indicator_v2(
                int32_or_64 *MatB_bit,
                float *dA_dense_gmem,
                int *group_id_gmem,
                int *spilled_row_hash_table_reverse_gmem,
                float *d_group_ele_row_val,
                int *spilled_row_cnt_offset,
                int *spilled_nnz_offset,
                float *tile_spilled_csrVal,
                int *tile_spilled_csrColInd,
                int *tile_spilled_csrRowPtr,
                int *dA_tiled_csr_offset_gmem,
                int *dA_tiled_csr_column_gmem,
                float *dA_tiled_csr_value_gmem,
                int *dA_tile_nnz_acc,
                int *dA_tile_nnz,
                float *final_result_gmem
            )
{
    // __shared__ float dA_dense_smem[TILE_HEIGHT][SPLIT_K];
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];
    __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ int32_or_64 MatB_bit_smem[SPLIT_K];

    __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];
    __shared__ float tiled_csr_value_smem[MAX_TILEA_NNZ];

    for (int i = 0; i < MAX_TILEA_NNZ; i++)
    {
        tiled_csr_column_smem[i] = 0;
        tiled_csr_value_smem[i] = 0;
    }
    
    int row_group_id;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;  
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    int group[MAX_GROUP_NUM];
    
    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        int tileA_id = gridDim.x * blockIdx.y + k;
        int tileB_id = k * gridDim.x + blockIdx.x;

        // Load MatA's csr data into shared memory
        for (int i = 0; i < TILE_HEIGHT+1; i++)
        {
            tiled_csr_offset_smem[i] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+i];
        }
        for (int i = 0; i < dA_tile_nnz[tileA_id]; i++)
        {
            tiled_csr_value_smem[i] = dA_tiled_csr_value_gmem[dA_tile_nnz_acc[tileA_id]+i];
            tiled_csr_column_smem[i] = dA_tiled_csr_column_gmem[dA_tile_nnz_acc[tileA_id]+i];
        }

        if (k == 0 && tid == 0 && bid == 0)
        {
            for (int i = 0; i < TILE_HEIGHT+1; i++)
            {
                printf("tiled_csr_offset_smem: %d, i: %d, bid: %d\n", tiled_csr_offset_smem[i], i, bid);
            }
            // for (int i = 0; i < dA_tile_nnz[tileA_id]; i++)
            // {
            //     printf("tiled_csr_column_smem: %d, i: %d, bid: %d\n", tiled_csr_column_smem[i], i, bid);
            // }
            // for (int i = 0; i < dA_tile_nnz[tileA_id]; i++)
            // {
            //     printf("tiled_csr_value_smem: %f, i: %d, bid: %d\n", tiled_csr_value_smem[i], i, bid);
            // }
        }


        // Load MatB's group data into shared memory
        if (k % 8 == 0)
        {
            for (int j = 0; j < 8; j++)
            {
                for (int i = 0; i < MAX_GROUP_NUM; i++)
                {
                    group[i] = d_group_ele_row_val[(MAX_GROUP_NUM * (j * gridDim.x + blockIdx.x) + i) * TILE_WIDTH + threadIdx.x];
                }
            }
        }

        // Load MatB's bit mask data into shared memory
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int z = 0; z < TILE_WIDTH; z++)
            {
                int entry_B = ((SPLIT_K * k) + z) * gridDim.x + blockIdx.x;
                MatB_bit_smem[z] = MatB_bit[entry_B];
            }
        }

        // Load MatB's group information into shared memory
        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        int rowB_ind = k * SPLIT_K + tid;
        int entry = rowB_ind * gridDim.x + blockIdx.x;
        group_id_smem[tid] = group_id_gmem[entry];
        spilled_row_hash_table_reverse_smem[tid] 
            = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        int rowA_ind = blockIdx.y * blockDim.x + threadIdx.x;
        for (int z = tiled_csr_offset_smem[threadIdx.x]; z < tiled_csr_offset_smem[threadIdx.x+1]; z++)
        {
            int entry_col = tiled_csr_column_smem[z];
            if (k == 0 && tid == 0 && bid == 0)
            {
                printf("entry_col: %d\n", entry_col);
            }
            int tmp = (__float_as_int(tiled_csr_value_smem[z]) >> threadIdx.y) & 1 == 0x01;
            // if ((__float_as_int(dA_dense_gmem[entry]) >> threadIdx.y) & 1 == 0x01)
            if ((threadIdx.y % 2 + entry_col % 2) % 2 == 0)
            {
                row_group_id = group_id_smem[entry_col];
                if (row_group_id != -1)
                {
                    atomicOr(&group_indicator[threadIdx.x][threadIdx.y][row_group_id], MatB_bit_smem[entry_col]);
                }
                else 
                {
                    // Current row_ind_in_tile in MatrixB is the spilled row
                    // Perform the extra computation
                    int row_in_csr = spilled_row_hash_table_reverse_smem[entry_col];
                    int start_offset;
                    if (row_in_csr == 0)
                    {
                        start_offset = 0;
                    }
                    else 
                    {
                        start_offset = tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                    }
                    for (int j = start_offset; j < tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                    {
                        int col_ind = tile_spilled_csrColInd[spilled_nnz_offset[tileB_id] + j];
                        result[threadIdx.x][threadIdx.y][col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
                    }
                }
            }
        }

        __syncthreads();

        for (int z = 0; z < TILE_HEIGHT; z++)
        {
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                if (((group_indicator[z][threadIdx.y][i] >> threadIdx.x) & 0x01) == 1)
                {
                    result[z][threadIdx.y][threadIdx.x] += group[i];
                    // result[threadIdx.x][threadIdx.y][z] += i;
                }
            }
        }
    }

    // compute with cuda core
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        int ind = (blockIdx.y * blockDim.x + threadIdx.x) * SIZE_N + blockIdx.x * TILE_WIDTH + i;
        final_result_gmem[ind] += result[threadIdx.x][threadIdx.y][i] * float(threadIdx.y);
    }

}

// template <typename int32_or_64>
// __global__ void spgemm_compute(
//             int32_or_64 group_indicator_gmem,
//             float *d_group_ele_row_val)
// {
//     __shared__ int32_or_64 group_indicator_smem[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
//     float group[MAX_GROUP_NUM];
//     for (int i = 0; i < MAX_GROUP_NUM; i++)
//     {
//         group[i] = d_group_ele_row_val[(MAX_GROUP_NUM * tileB_id + i) * TILE_WIDTH + threadIdx.x];
//     }

//     for (int i = 0; i < TILE_HEIGHT; i++) 
//     {
//         for (int b = 0; b < BIT_WIDTH; b++)
//         {
//             for (int j = 0; j < MAX_GROUP_NUM; j++) 
//             {
//                 register_idx = (group_indicator[i][b][j] >> threadIdx.x) & 1;
//                 if (register_idx == 1)
//                 {
//                     result[i][b][threadIdx.x] += group[1 << j];
//                 }
//             }
//         }
//     }
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
                                int *spilled_row_cnt_offset,
                                int *spilled_nnz_offset,
                                float *tile_spilled_csrVal,
                                int *tile_spilled_csrColInd,
                                int *tile_spilled_csrRowPtr,
                                int *spilled_row_hash_table_reverse_gmem,
                                float *final_result_gmem
                                )
{
    // int bid = blockIdx.y * gridDim.x + blockIdx.x;
    int assigned_row_ind = blockIdx.y * blockDim.x + threadIdx.x;
    // int assigned_col_ind = blockIdx.x * split_k;
    // int assigned_bit_pos = threadIdx.x % BIT_WIDTH;
    // int entry_ind_bit = assigned_row_ind * gridDim.x + blockIdx.x;

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
                    // int mv_bit = b+16;
                    // if((tmp >> mv_bit) & 1 == 0x01)
                    // TEST!!! UNCOMMENT THE ABOVE TWO LINES!!
                    if ((b%2+row_ind_in_tile%2)%2 == 0)
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
                            int row_in_csr = spilled_row_hash_table_reverse_smem[row_ind_in_tile];
                            int start_offset;
                            if (row_in_csr == 0)
                            {
                                start_offset = 0;
                            }
                            else 
                            {
                                start_offset = tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                            }
                            for (int j = start_offset; j < tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                            {
                                col_ind = tile_spilled_csrColInd[spilled_nnz_offset[tileB_id] + j];
                                result[threadIdx.x][b][col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
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
                    register_idx = (group_indicator[i][b][j] >> threadIdx.x) & 1;
                    if (register_idx == 1)
                    {
                        result[i][b][threadIdx.x] += group[1 << j];
                    }
                }
            }
        }
    }

    // compute with cuda kernel
    for (int i = 0; i < blockDim.x; i++) 
    {
        for (int b = 0; b < BIT_WIDTH; b++)
        {
            int ind = (blockIdx.y * blockDim.x + i) * SIZE_N + blockIdx.x * TILE_WIDTH + threadIdx.x;
            final_result_gmem[ind] += result[i][b][threadIdx.x] * float(b);
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

    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
    const int m = SIZE_M;
    const int k = SIZE_K;
	const int n = SIZE_N;

    int tileB_cnt = k * n / SPLIT_K / TILE_WIDTH;
    int tileB_x_cnt = n / TILE_WIDTH;
    int tileB_y_cnt = k / SPLIT_K;

    dim3 grid1(tileB_x_cnt, tileB_y_cnt, 1), block1(SPLIT_K, 1, 1);
    dim3 grid2(SIZE_M/TILE_HEIGHT, SIZE_N/TILE_WIDTH, 1), block2(TILE_HEIGHT, 1, 1);

	float* hA_dense = (float*)malloc(sizeof(float)*m*k);
    float* hB_dense = (float*)malloc(sizeof(float)*k*n);
    float* hC_dense = (float*)malloc(sizeof(float)*m*n);
    fill_random(hA_dense, m, k, SPARSITY);
    fill_random(hB_dense, k, n, SPARSITY);
    fill_random(hC_dense, m, n, SPARSITY);

    // basic ptrs
    float *dA_dense, *dA_csr_values;
    float *dB_dense, *dB_csr_values;
    int   *dA_csr_offsets, *dA_csr_columns;
    int   *dB_csr_offsets, *dB_csr_columns;
    int   *dC_csrOffsets, *dC_columns;
    float *dC_values;

    // advanced ptrs
    float *dB_group_ele_val;
    int *dB_group_id, *dB_spilled_row_cnt, *dB_spilled_nnz;
    int *dB_spilled_row_hash_table_gmem, *dB_spilled_row_hash_table_reverse_gmem;
    int *dB_group_ele_ind;
    int *dB_bitmask, *dB_groupmask;

    // basic allocation
    CHECK_CUDA( cudaMalloc((void**) &dA_dense,          m * k * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_offsets,   (m + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_dense,          k * n * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_offsets,   (k + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_csrOffsets,    (m + 1) * sizeof(int)) )

    // advanced allocation
    CHECK_CUDA( cudaMalloc((void**) &dB_bitmask,        k * n / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_groupmask,      tileB_cnt * MAX_GROUP_NUM * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_ele_ind,  k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_ele_val,  k * n / SPLIT_K * MAX_GROUP_NUM * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_group_id,       k * n / TILE_WIDTH * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_cnt,tileB_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_nnz,    tileB_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_hash_table_gmem,
                                    tileB_cnt * SPLIT_K * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_hash_table_reverse_gmem,
                                   tileB_cnt * SPLIT_K * sizeof(int)) )

    
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
    int64_t num_rows_tmpA, num_cols_tmpA;
    // dense2CSR(m, k, dA_dense, dA_csr_values, dA_csr_offsets, dA_csr_columns, nnzA);

    int num_colsA = k;
    int num_rowsA = m;
    int ld = num_colsA;
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA_sp;
    cusparseDnMatDescr_t matA_dn;
    void*                dBufferA    = NULL;
    size_t               bufferSizeA = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA_dn, num_rowsA, num_colsA, ld, dA_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA_sp, num_rowsA, num_colsA, 0,
                                      dA_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA_dn, matA_sp,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSizeA) )
    CHECK_CUDA( cudaMalloc(&dBufferA, bufferSizeA) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA_dn, matA_sp,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBufferA) )
    // get number of non-zero elements
    CHECK_CUSPARSE( cusparseSpMatGetSize(matA_sp, &num_rows_tmpA, &num_cols_tmpA,
                                         &nnzA) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_columns, nnzA * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dA_csr_values,  nnzA * sizeof(float)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matA_sp, dA_csr_offsets, dA_csr_columns,
                                           dA_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA_dn, matA_sp,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBufferA) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA_dn) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA_sp) )
    // CHECK_CUSPARSE( cusparseDestroy(handle) )

    printf("Transform CSR to tiled CSR\n");
    int tileA_cnt = (SIZE_M/TILE_HEIGHT)*(SIZE_K/SPLIT_K);
    int *dA_tiled_csr_offset, *dA_tiled_csr_column;
    int *dA_tile_nnz_acc, *dA_tile_nnz, *dA_tile_row_nnz;
    float *dA_tiled_csr_value;
    CHECK_CUDA( cudaMalloc((void**) &dA_tile_nnz,         sizeof(int) * tileA_cnt) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tile_nnz_acc,     sizeof(int) * (tileA_cnt+1)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tile_row_nnz,     sizeof(int) * SIZE_M * SIZE_K / SPLIT_K) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tiled_csr_offset, sizeof(int) * tileA_cnt * (TILE_HEIGHT+1)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tiled_csr_column, sizeof(int) * nnzA) )
    CHECK_CUDA( cudaMalloc((void**) &dA_tiled_csr_value,  sizeof(float) * nnzA) )

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
    int64_t num_rows_tmpB, num_cols_tmpB;
    // dense2CSR(m, k, dA_dense, dA_csr_values, dA_csr_offsets, dA_csr_columns, nnzA);

    int num_colsB = n;
    int num_rowsB = k;
    ld = num_colsB;
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseSpMatDescr_t matB_sp;
    cusparseDnMatDescr_t matB_dn;
    void*                dBufferB    = NULL;
    size_t               bufferSizeB = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB_dn, num_rowsB, num_colsB, ld, dB_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB_sp, num_rowsB, num_colsB, 0,
                                      dB_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matB_dn, matB_sp,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSizeB) )
    CHECK_CUDA( cudaMalloc(&dBufferB, bufferSizeB) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matB_dn, matB_sp,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBufferB) )
    // get number of non-zero elements
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB_sp, &num_rows_tmpB, &num_cols_tmpB,
                                         &nnzB) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_columns, nnzB * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dB_csr_values,  nnzB * sizeof(float)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB_sp, dB_csr_offsets, dB_csr_columns,
                                           dB_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matB_dn, matB_sp,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBufferB) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB_dn) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB_sp) )

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

    int *hB_spilled_nnz = (int*)malloc(tileB_cnt * sizeof(int));
    int *hB_spilled_row_cnt = (int*)malloc(tileB_cnt * sizeof(int));
    cudaMemcpy(hB_spilled_nnz, dB_spilled_nnz, tileB_cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_spilled_row_cnt, dB_spilled_row_cnt, tileB_cnt * sizeof(int), cudaMemcpyDeviceToHost);

    int nnz_cnt = 0;
    int row_cnt = 0;
    int *hB_spilled_nnz_offset = (int*)malloc(tileB_cnt * sizeof(int));
    int *hB_spilled_row_cnt_offset = (int*)malloc(tileB_cnt * sizeof(int));
    float *dB_tile_spilled_csrVal;
    int *dB_tile_spilled_csrColInd, *dB_tile_spilled_csrRowPtr;
    int *dB_spilled_nnz_offset, *dB_spilled_row_cnt_offset;
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_nnz_offset,     tileB_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_spilled_row_cnt_offset,  tileB_cnt * sizeof(int)) )
    for (int i = 0; i < tileB_cnt; i++)
    {
        std::cout << "hB_spilled_nnz     -- " << i << ": " << hB_spilled_nnz[i] << std::endl;
        std::cout << "hB_spilled_row_cnt -- " << i << ": " << hB_spilled_row_cnt[i] << std::endl;
        hB_spilled_nnz_offset[i] = nnz_cnt;
        hB_spilled_row_cnt_offset[i] = row_cnt;
        nnz_cnt += hB_spilled_nnz[i];
        row_cnt += hB_spilled_row_cnt[i];
    }
    cudaMemcpy(dB_spilled_nnz_offset, hB_spilled_nnz_offset, tileB_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_spilled_row_cnt_offset, hB_spilled_row_cnt_offset, tileB_cnt * sizeof(int), cudaMemcpyHostToDevice);

    CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrVal,     nnz_cnt * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrColInd,  nnz_cnt * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dB_tile_spilled_csrRowPtr,  row_cnt * sizeof(int)) )

    generate_spilled_csr<<<grid1, block1>>>(dB_bitmask,                            // input
                                     dB_groupmask,                          // output, for visualization
                                     dB_group_ele_ind,                      // output, not necessary
                                     dB_group_ele_val,                      // output
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
    float *hB_tile_spilled_csrVal = (float*)malloc(nnz_cnt * sizeof(float));
    int *hB_tile_spilled_csrColInd = (int*)malloc(nnz_cnt * sizeof(int));
    int *hB_tile_spilled_csrRowPtr = (int*)malloc(row_cnt * sizeof(int));

    cudaMemcpy(hB_tile_spilled_csrVal, dB_tile_spilled_csrVal, nnz_cnt * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_tile_spilled_csrColInd, dB_tile_spilled_csrColInd, nnz_cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB_tile_spilled_csrRowPtr, dB_tile_spilled_csrRowPtr, row_cnt * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < nnz_cnt; i++)
    // {
    //     std::cout << "ColInd: " << hB_tile_spilled_csrColInd[i] << "   Value: " << hB_tile_spilled_csrVal[i] << std::endl;
    // }

    // int *hB_groupmask = (int*)malloc(k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int));
    // cudaMemcpy(hB_groupmask, dB_groupmask, k * n / SPLIT_K / TILE_WIDTH * MAX_GROUP_NUM * sizeof(int), cudaMemcpyDeviceToHost);
    // printintMatrix_32(16, hB_groupmask, "B_groupmask");

    float *dC_final_result_gmem;
    CHECK_CUDA( cudaMalloc((void**) &dC_final_result_gmem,  SIZE_M * SIZE_N * sizeof(float)) )

    cudaEvent_t start, end, cusparse_start, cusparse_end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&cusparse_start);
    cudaEventCreate(&cusparse_end);

    // spgemm

    cudaEventRecord(start);
    // bit_wise_spgemm<<<grid2, block2>>>(SPLIT_K, 
    //                                 dA_csr_values, 
    //                                 dA_csr_offsets, 
    //                                 dA_csr_columns, 
    //                                 dB_group_ele_val, 
    //                                 dB_bitmask, 
    //                                 dB_group_id, 
    //                                 dB_spilled_row_cnt,
    //                                 dB_spilled_nnz,
    //                                 dB_spilled_row_cnt_offset,
    //                                 dB_spilled_nnz_offset,
    //                                 dB_tile_spilled_csrVal,
    //                                 dB_tile_spilled_csrColInd,
    //                                 dB_tile_spilled_csrRowPtr,
    //                                 dB_spilled_row_hash_table_reverse_gmem,
    //                                 dC_final_result_gmem
    //                                 );
    printf("Generate group indicator\n");
    dim3 grid3(SIZE_N/TILE_WIDTH, SIZE_M/TILE_HEIGHT, 1), block3(TILE_HEIGHT, BIT_WIDTH, 1);
    generate_group_indicator_v2<<<grid3, block3>>>(dB_bitmask, 
                                                dA_dense, 
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
                                                dA_tile_nnz,
                                                dC_final_result_gmem
                                                );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // spgemm_compute<<<grid2, block2>>>(dA_group_indicator_t_gmem, dB_group_ele_val);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    printf("Elapsed time1: %fms\n", ms);

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

    cudaEventRecord(cusparse_end);
    cudaEventSynchronize(cusparse_end);

    float cusparse_ms;
    cudaEventElapsedTime(&cusparse_ms, cusparse_start, cusparse_end);
    cudaEventDestroy(cusparse_start);
    cudaEventDestroy(cusparse_end);

    printf("Elapsed time2: %fms\n", cusparse_ms);

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

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )


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
        // for (int i = 0; i < 32; i++)
        // {
        //     std::cout << std::left << std::setw(4) << hB_group_ele_ind[i] << std::endl;
        // }
    }
    
    // size_t *size;
    // cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
    // std::cout << "HeapSize: " << *size << std::endl;

    // free(dB)

    // std::cout << "Input matrix A has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n";
    // std::cout << "             B has shape (" << B.num_rows << "," << B.num_cols << ") and " << B.num_entries << " entries" << "\n\n";

}