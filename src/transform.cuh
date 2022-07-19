#ifndef _TRANSFORM_
#define _TRANSFORM_
#include "common.h"

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
        int tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        int tileB_id = k * gridDim.x + blockIdx.x;

        int tile_nnz = dA_tile_nnz[tileA_id];
        int tile_nnz_acc = dA_tile_nnz_acc[tileA_id];


        // Load MatA's csr data into shared memory
        if (tid == 0)
        {
            for (int i = 0; i < TILE_HEIGHT+1; i++)
            {
                tiled_csr_offset_smem[i] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+i];
            }
            for (int i = 0; i < tile_nnz; i++)
            {
                tiled_csr_value_smem[i] = dA_tiled_csr_value_gmem[tile_nnz_acc+i];
                tiled_csr_column_smem[i] = dA_tiled_csr_column_gmem[tile_nnz_acc+i];
            }
        }

        __syncthreads();


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

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        int rowB_ind = k * SPLIT_K + tid;
        int entry = rowB_ind * gridDim.x + blockIdx.x;

        // Load MatB's group information into shared memory
        group_id_smem[tid] = group_id_gmem[entry];

        // Load MatB's bit mask data into shared memory
        MatB_bit_smem[tid] = MatB_bit[entry];

        spilled_row_hash_table_reverse_smem[tid] 
            = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        int rowA_ind = blockIdx.y * blockDim.x + threadIdx.x;
        for (int z = tiled_csr_offset_smem[threadIdx.x]; z < tiled_csr_offset_smem[threadIdx.x+1]; z++)
        {
            int entry_col = tiled_csr_column_smem[z];
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


template <typename int32_or_64>
__global__ void generate_group_indicator_v3(
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
    
    int row_group_id;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;  
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    int group[MAX_GROUP_NUM];
    
    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        int tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        int tileB_id = k * gridDim.x + blockIdx.x;

        int tile_nnz = dA_tile_nnz[tileA_id];
        int tile_nnz_acc = dA_tile_nnz_acc[tileA_id];


        // Load MatA's csr data into shared memory
        if (tid == 0)
        {
            for (int i = 0; i < TILE_HEIGHT+1; i++)
            {
                tiled_csr_offset_smem[i] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+i];
            }
            for (int i = 0; i < tile_nnz; i++)
            {
                tiled_csr_value_smem[i] = dA_tiled_csr_value_gmem[tile_nnz_acc+i];
                tiled_csr_column_smem[i] = dA_tiled_csr_column_gmem[tile_nnz_acc+i];
            }
        }

        __syncthreads();


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

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        int rowB_ind = k * SPLIT_K + tid;
        int entry = rowB_ind * gridDim.x + blockIdx.x;

        // Load MatB's group information into shared memory
        group_id_smem[tid] = group_id_gmem[entry];

        // Load MatB's bit mask data into shared memory
        MatB_bit_smem[tid] = MatB_bit[entry];

        spilled_row_hash_table_reverse_smem[tid] 
            = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        for (int z = tiled_csr_offset_smem[threadIdx.x]; z < tiled_csr_offset_smem[threadIdx.x+1]; z++)
        {
            int entry_col = tiled_csr_column_smem[z];
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


#endif