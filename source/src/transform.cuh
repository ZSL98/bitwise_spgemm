#ifndef _TRANSFORM_
#define _TRANSFORM_
#include "common.h"
#include <math.h>

/*
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
*/

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
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

        // Load MatB's group data into shared memory
        if (k % 8 == 0)
        {
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                group[i] = d_group_ele_row_val[(MAX_GROUP_NUM * (threadIdx.y * gridDim.x + blockIdx.x) + i) * TILE_WIDTH + threadIdx.x];
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
    // __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ int32_or_64 MatB_bit_smem[SPLIT_K];

    // __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    // __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];
    // __shared__ float tiled_csr_value_smem[MAX_TILEA_NNZ];
    
    int row_group_id;
    // int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    int group[MAX_GROUP_NUM];
    int MatA_bit[8];
    // int32_or_64 group_indicator[MAX_GROUP_NUM];
    float result[TILE_WIDTH];
    
    // for (int i = 0; i < MAX_GROUP_NUM; i++)
    // {
    //     group_indicator[i] = 0;
    // }

    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        int tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        int tileB_id = k * gridDim.x + blockIdx.x;

        int tile_nnz = dA_tile_nnz[tileA_id];
        int tile_nnz_acc = dA_tile_nnz_acc[tileA_id];

        // tiled_csr_value_smem[tid] = dA_tiled_csr_value_gmem[tile_nnz_acc+tid];
        // Load MatA's csr data into shared memory
        for (int i = 0; i < 8; i++)
        {
            MatA_bit[i] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+i];
        }

        // // Load MatA's csr data into shared memory
        // if (tid == 0)
        // {
        //     for (int i = 0; i < TILE_HEIGHT+1; i++)
        //     {
        //         tiled_csr_offset_smem[i] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+i];
        //     }
        //     for (int i = 0; i < tile_nnz; i++)
        //     {
        //         tiled_csr_value_smem[i] = dA_tiled_csr_value_gmem[tile_nnz_acc+i];
        //         tiled_csr_column_smem[i] = dA_tiled_csr_column_gmem[tile_nnz_acc+i];
        //     }
        // }

        // tiled_csr_value_reg = tiled_csr_column_smem[tid];

        // Load MatB's group data from global memory into registers
        if (k % 8 == 0)
        {
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                group[i] = d_group_ele_row_val[(MAX_GROUP_NUM * (threadIdx.y * gridDim.x + blockIdx.x) + i) * TILE_WIDTH + threadIdx.x];
            }
        }

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        int rowB_ind = k * SPLIT_K + tid;
        int entry = rowB_ind * gridDim.x + blockIdx.x;

        group_id_smem[tid] = group_id_gmem[entry];  // Load MatB's group information into shared memory
        MatB_bit_smem[tid] = MatB_bit[entry];       // Load MatB's b it mask data into shared memory
        spilled_row_hash_table_reverse_smem[tid] = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        for (int z = 0; z < 8; z++)
        {
            // int entry_col = MatA_bit[z];
            // int MatB_bit_reg = MatB_bit_smem[entry_col];
            for (int i = 0; i < 32; i++)
            {
                // int tmp = (__float_as_int(tiled_csr_value_smem[z]) >> threadIdx.y) & 1 == 0x01;
                // if ((__float_as_int(dA_dense_gmem[entry]) >> threadIdx.y) & 1 == 0x01)
                // if ((threadIdx.y % 2 + entry_col % 2) % 2 == 0)
                if ((MatA_bit[z] >> i) == 0x01)
                {
                    row_group_id = group_id_smem[i];
                    if (row_group_id != -1)
                    {
                        atomicOr(&group_indicator[threadIdx.x][threadIdx.y][row_group_id], MatB_bit_smem[i]);
                    }
                    else 
                    {
                        // Current row_ind_in_tile in MatrixB is the spilled row
                        // Perform the extra computation
                        int row_in_csr = spilled_row_hash_table_reverse_smem[i];
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
                            // result[threadIdx.x][threadIdx.y][col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
                            result[col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
                        }
                    }
                }
            }
        }

        __syncthreads();

        // for (int z = 0; z < TILE_HEIGHT; z++)
        // {
        //     for (int i = 0; i < MAX_GROUP_NUM; i++)
        //     {
        //         if (((group_indicator[z][threadIdx.y][i] >> threadIdx.x) & 0x01) == 1)
        //         {
        //             result[z][threadIdx.y][threadIdx.x] += group[i];
        //             // result[threadIdx.x][threadIdx.y][z] += i;
        //         }
        //     }
        // }

        for (int z = 0; z < TILE_WIDTH; z++)
        {
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                if (((group_indicator[threadIdx.x][threadIdx.y][i] >> z) & 0x01) == 1)
                {
                    result[z] += group[i];
                    // result[threadIdx.x][threadIdx.y][z] += i;
                }
            }
        }

    }

    // compute with cuda core
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        int ind = (blockIdx.y * blockDim.x + threadIdx.x) * SIZE_N + blockIdx.x * TILE_WIDTH + i;
        // final_result_gmem[ind] += result[threadIdx.x][threadIdx.y][i] * float(threadIdx.y);
        final_result_gmem[ind] += result[i] * float(threadIdx.y);
    }

}


template <typename int32_or_64>
__global__ void generate_group_indicator_smem_dense(
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
    __shared__ int group_id_smem[2][SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[2][SPLIT_K];
    __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    __shared__ int32_or_64 MatB_bit_smem[2][SPLIT_K];

    // __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    // __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];
    // __shared__ float tiled_csr_value_smem[MAX_TILEA_NNZ];
    
    int row_group_id;
    // int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    int group[MAX_GROUP_NUM];
    int MatA_bit[8];
    // int32_or_64 group_indicator[MAX_GROUP_NUM];
    // float result[TILE_WIDTH];
    
    // for (int i = 0; i < MAX_GROUP_NUM; i++)
    // {
    //     group_indicator[i] = 0;
    // }

    int entry = tid * gridDim.x + blockIdx.x;
    group_id_smem[0][tid] = group_id_gmem[entry];  // Load MatB's group information into shared memory
    MatB_bit_smem[0][tid] = MatB_bit[entry];       // Load MatB's b it mask data into shared memory
    spilled_row_hash_table_reverse_smem[0][tid] = spilled_row_hash_table_reverse_gmem[tid + blockIdx.x * SPLIT_K];


    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        // Load MatA's data into registers
        int tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        for (int i = 0; i < 8; i++)
        {
            MatA_bit[i] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+i];
        }

        // Load MatB's group data from global memory into registers
        if (k % 8 == 0)
        {
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                group[i] = d_group_ele_row_val[(MAX_GROUP_NUM * (threadIdx.y * gridDim.x + blockIdx.x) + i) * TILE_WIDTH + threadIdx.x];
            }
        }

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        int rowB_ind = k * SPLIT_K + tid;
        int entry = rowB_ind * gridDim.x + blockIdx.x;
        int tileB_id = (k+1) * gridDim.x + blockIdx.x;

        group_id_smem[(k+1)%2][tid] = group_id_gmem[entry];  // Load MatB's group information into shared memory
        MatB_bit_smem[(k+1)%2][tid] = MatB_bit[entry];       // Load MatB's b it mask data into shared memory
        spilled_row_hash_table_reverse_smem[(k+1)%2][tid] = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        // #pragma unroll
        for (int z = 0; z < SPLIT_K; z++)
        {
            // int tmp = (__float_as_int(tiled_csr_value_smem[z]) >> threadIdx.y) & 1 == 0x01;
            // if ((__float_as_int(dA_dense_gmem[entry]) >> threadIdx.y) & 1 == 0x01)
            // if ((threadIdx.y % 2 + entry_col % 2) % 2 == 0)
            int m = z % TILE_WIDTH;
            int n = z / TILE_WIDTH;
            if ((MatA_bit[n] >> m) == 0x01)
            {
                row_group_id = group_id_smem[k%2][z];
                if (row_group_id != -1)
                {
                    atomicOr(&group_indicator[threadIdx.x][threadIdx.y][row_group_id], MatB_bit_smem[k%2][z]);
                }
                else 
                {
                    // Current row_ind_in_tile in MatrixB is the spilled row
                    // Perform the extra computation
                    int row_in_csr = spilled_row_hash_table_reverse_smem[k%2][m];
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
                        // result[col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
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
        // final_result_gmem[ind] += result[i] * float(threadIdx.y);
    }

}


template <typename int32_or_64>
__global__ void generate_group_indicator_smem_sparse(
                int32_or_64 *MatB_bit,
                float *dA_dense_gmem,
                float *dB_dense_gmem,
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
    // input buffers
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];
    __shared__ int32_or_64 MatB_bit_smem[SPLIT_K];
    __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];
    __shared__ float tiled_csr_value_smem[MAX_TILEA_NNZ];
    __shared__ float group[TILE_WIDTH][MAX_GROUP_NUM];

    // intermediate buffers
    // __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    // __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    float result[TILE_WIDTH];
    int32_or_64 group_indicator[MAX_GROUP_NUM];
    int32_or_64 or_group_indicator = 0;

    // for (int i = 0; i < MAX_TILEA_NNZ; i++)
    // {
    //     tiled_csr_column_smem[i] = 0;
    //     tiled_csr_value_smem[i] = 0;
    // }
    
    int row_group_id;
    // int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    // int group[MAX_GROUP_NUM];

    // // Declare the fragments
    // nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    // nvcuda::wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    // nvcuda::wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // // Initialize the output to zero
    // nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // // Load the inputs
    // nvcuda::wmma::load_matrix_sync(a_frag, (half *)&group[0][0], 16);
    // nvcuda::wmma::load_matrix_sync(b_frag, (half *)&group[0][0], 16);

    // // Perform the matrix multiplication
    // nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // // Store the output
    // nvcuda::wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);

    int tmp;
    int rowB_ind;
    int entry;
    int col_ind, ind, entry_col;
    int tileA_id, tileB_id, tile_nnz_acc;
    
    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        tileB_id = k * gridDim.x + blockIdx.x;
        tile_nnz_acc = dA_tile_nnz_acc[tileA_id];
        // int tile_nnz = dA_tile_nnz[tileA_id];

        // Load MatA's csr data into shared memory
        if (threadIdx.y == 0)
        {
            tiled_csr_offset_smem[threadIdx.x] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+threadIdx.x];
        }
        if (tid == 0)
        {
            tiled_csr_offset_smem[TILE_HEIGHT] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+TILE_HEIGHT];
        }
        for (int i = 0; i < MAX_TILEA_NNZ/256; i++)
        {
            tiled_csr_value_smem[tid + i*256] = dA_tiled_csr_value_gmem[tile_nnz_acc + tid + i*256];
            tiled_csr_column_smem[tid + i*256] = dA_tiled_csr_column_gmem[tile_nnz_acc + tid + i*256];
        }

        // // Load MatB's group data into registers
        // if (k % 8 == 0)
        // {
        //     for (int i = 0; i < MAX_GROUP_NUM; i++)
        //     {
        //         group[i] = d_group_ele_row_val[(MAX_GROUP_NUM * (threadIdx.y * gridDim.x + blockIdx.x) + i) * TILE_WIDTH + threadIdx.x];
        //     }
        // }

        // Load MatB's group data into shared memory
        group[threadIdx.x][threadIdx.y] = d_group_ele_row_val[(MAX_GROUP_NUM * tileB_id + threadIdx.y) * TILE_WIDTH + threadIdx.x];

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        rowB_ind = k * SPLIT_K + tid;
        entry = rowB_ind * gridDim.x + blockIdx.x;

        // Load MatB's group information into shared memory
        group_id_smem[tid] = group_id_gmem[entry];

        // Load MatB's bit mask data into shared memory
        MatB_bit_smem[tid] = MatB_bit[entry];

        spilled_row_hash_table_reverse_smem[tid] 
            = spilled_row_hash_table_reverse_gmem[tid + tileB_id * SPLIT_K];

        __syncthreads();

        // int rowA_ind = blockIdx.y * blockDim.x + threadIdx.x;
        for (int z = tiled_csr_offset_smem[threadIdx.x]; z < tiled_csr_offset_smem[threadIdx.x+1]; z++)
        {
            entry_col = tiled_csr_column_smem[z];
            tmp = (__float_as_int(tiled_csr_value_smem[z]) >> threadIdx.y) & 1 == 0x01;
            // if ((__float_as_int(dA_dense_gmem[entry]) >> threadIdx.y) & 1 == 0x01)
            if ((threadIdx.y % 2 + entry_col % 2) % 2 == 0)
            // if (tmp)
            {
                row_group_id = group_id_smem[entry_col];
                if (row_group_id != -1)
                {
                    group_indicator[row_group_id] |= MatB_bit_smem[entry_col];
                    // atomicOr(&group_indicator[threadIdx.x][threadIdx.y][row_group_id], MatB_bit_smem[entry_col]);
                }
                else 
                {
                    // Current row_ind_in_tile in MatrixB is the spilled row
                    // Perform the extra computation
                    for (int i = 0; i < TILE_WIDTH; i++)
                    {
                        result[i] += dB_dense_gmem[(k * SPLIT_K + entry_col) * SIZE_N + blockIdx.x * TILE_WIDTH + i];
                    }
                    // int row_in_csr = spilled_row_hash_table_reverse_smem[entry_col];
                    // int start_offset;
                    // if (row_in_csr == 0)
                    // {
                    //     start_offset = 0;
                    // }
                    // else 
                    // {
                    //     start_offset = tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                    // }
                    // for (int j = start_offset; j < tile_spilled_csrRowPtr[spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                    // {
                    //     col_ind = tile_spilled_csrColInd[spilled_nnz_offset[tileB_id] + j];
                    //     // result[threadIdx.x][threadIdx.y][col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
                    //     result[col_ind] += tile_spilled_csrVal[spilled_nnz_offset[tileB_id] + j];
                    // }
                }
            }
        }

        __syncthreads();

        // for (int z = 0; z < TILE_HEIGHT; z++)
        // {
        //     for (int i = 0; i < MAX_GROUP_NUM; i++)
        //     {
        //         if (((group_indicator[z][threadIdx.y][i] >> threadIdx.x) & 0x01) == 1)
        //         {    
        //             result[z][threadIdx.y][threadIdx.x] += group[i];
        //             // result[threadIdx.x][threadIdx.y][z] += i;
        //         }
        //     }
        // }

        #pragma unroll
        for (int i = 0; i < MAX_GROUP_NUM; i++)
        {
            if (group_indicator[i] == 0)
            {
                continue;
            }
            #pragma unroll
            for (int z = 0; z < TILE_WIDTH; z++)
            {
                if (((group_indicator[i] >> z) & 0x01) == 1)
                {
                    result[z] += group[z][i];
                    // result[threadIdx.x][threadIdx.y][z] += group[z][i];
                    // result[threadIdx.x][threadIdx.y][z] += i;
                }
            }
        }
    }

    // compute with cuda core
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        ind = (blockIdx.y * blockDim.x + threadIdx.x) * SIZE_N + blockIdx.x * TILE_WIDTH + i;
        final_result_gmem[ind] += result[i] * float(threadIdx.y);
        // final_result_gmem[ind] += result[threadIdx.x][threadIdx.y][i] * float(threadIdx.y);
    }

}


__global__ void why(int *dC_bitmask)
{
    dC_bitmask[0] = 61;
}

template <typename BitMaskType>
__global__ void pre_spgemm(
                        BitMaskType *MatB_bit,
                        int *dC_spilled_row_cnt,
                        int *dC_spilled_nnz,
                        int *dA_tiled_csr_offset_gmem,
                        int *dA_tiled_csr_column_gmem,
                        int *dA_tile_nnz_acc,
                        int *dC_output_group_idx,
                        BitMaskType *dC_bitmask,
                        BitMaskType *dC_groupmask,
                        int *dC_spilled_row_row_idx,
                        int *dC_spilled_row_tile_idx,
                        int *dC_spilled_row_cnt_offset,
                        int *dC_spilled_nnz_offset,
                        int *dC_spilled_row_buffersize,
                        int *dC_spilled_nnz_buffersize,
                        int *nnz
                        )
{
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = bid * blockDim.x + threadIdx.x;
    // input buffers
    __shared__ BitMaskType MatB_bit_smem[SPLIT_K];
    __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];

    __shared__ int output_row_group[OUTPUT_MAX_GROUP_NUM];

    // Initialize
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < OUTPUT_MAX_GROUP_NUM; i++)
        {
            output_row_group[i] = 0;
        }
    }
    
    int rowB_ind;
    int entry;
    int col_ind, ind, entry_col;
    int tileA_id, tileB_id, tile_nnz_acc;
    BitMaskType bit_indicator = 0;

    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        tileB_id = k * gridDim.x + blockIdx.x;
        tile_nnz_acc = dA_tile_nnz_acc[tileA_id];

        tiled_csr_offset_smem[threadIdx.x] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+threadIdx.x];
        if (threadIdx.x == 0)
        {
            tiled_csr_offset_smem[TILE_HEIGHT] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+TILE_HEIGHT];
        }
        for (int i = 0; i < MAX_TILEA_NNZ/blockDim.x; i++)
        {
            tiled_csr_column_smem[threadIdx.x + i*blockDim.x] = dA_tiled_csr_column_gmem[tile_nnz_acc + threadIdx.x + i*blockDim.x];
        }

        rowB_ind = k * SPLIT_K + threadIdx.x;
        entry = rowB_ind * gridDim.x + blockIdx.x;
        // Load MatB's bit mask data into shared memory
        MatB_bit_smem[threadIdx.x] = MatB_bit[entry];
        __syncthreads();

        // printf("tiled_csr_offset_smem: %i, threadIdx.x: %i\n", tiled_csr_offset_smem[threadIdx.x], threadIdx.x);
        for (int z = tiled_csr_offset_smem[threadIdx.x]; z < tiled_csr_offset_smem[threadIdx.x+1]; z++)
        {
            entry_col = tiled_csr_column_smem[z];
            // printf("entry_col: %i\n", entry_col);
            bit_indicator |= MatB_bit_smem[entry_col];
        }
    }

    __syncthreads();

    dC_bitmask[tid] = bit_indicator;
    // atomicAdd(nnz, __popc(bit_indicator));

    // printf("bit_indicator: %i\n", bit_indicator);
    int output_group_idx = 0;
    BitMaskType and_result;
    BitMaskType expected = output_row_group[output_group_idx];
    and_result = expected & bit_indicator;
    while (and_result != 0)
    {
        output_group_idx++;
        expected = output_row_group[output_group_idx];
        and_result = expected & bit_indicator;
    }
    BitMaskType or_result = expected | bit_indicator;
    BitMaskType old_value = atomicCAS(&output_row_group[output_group_idx], expected, or_result);

    int spilled_idx;

    // For rows that haven't been added onto the row_group
    while (expected != old_value) {
        // calculate and_result again to see if there exists overlap
        expected = output_row_group[output_group_idx];
        and_result = expected & bit_indicator;
        // If there exists overlap, change to next row_group until no overlap exists
        while (and_result != 0) {
            output_group_idx++;
            if (output_group_idx >= OUTPUT_MAX_GROUP_NUM)
            {
                output_group_idx = -1;
                int spilled_row_hash_key = atomicAdd(&dC_spilled_row_cnt[bid], 1);
                // spilled_row_hash_table_smem[spilled_row_hash_key] = threadIdx.x;
                for (int j = 0; j < TILE_WIDTH; j++)
                {
                    if (((bit_indicator >> j) & 0x01) == 1)
                    {
                        spilled_idx = atomicAdd(&dC_spilled_nnz[bid], 1);
                    }
                }
                dC_spilled_row_row_idx[spilled_idx] = threadIdx.x;
                dC_spilled_row_tile_idx[spilled_idx] = bid;
                break;
            }
            expected = output_row_group[output_group_idx];
            and_result = expected & bit_indicator;
        }
        if (output_group_idx == -1)
        {
            break;
        }
        // expected = output_row_group[output_group_idx];
        // Now there is no overlap, try to add onto the new row_group.
        or_result = expected | bit_indicator;
        old_value = atomicCAS(&output_row_group[output_group_idx], expected, or_result);
    }
    dC_output_group_idx[tid] = output_group_idx;

    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < OUTPUT_MAX_GROUP_NUM; i++)
        {
            atomicAdd(nnz, __popc(output_row_group[i]));
        }
        // printf("bid: %d\n", bid);
        for (int i = 0; i < OUTPUT_MAX_GROUP_NUM; i++)
        {
            dC_groupmask[bid * OUTPUT_MAX_GROUP_NUM + i] = output_row_group[i];
            // dC_groupmask[bid * OUTPUT_MAX_GROUP_NUM + i] = 61;
        }
    }

    // printf("What happened? %d\n", dC_groupmask[0]);

    __syncthreads();
    if (tid == 0)
    {
        int tmp_row_cnt_size = 0;
        int tmp_nnz_size = 0;
        for (int i = 0; i < (SIZE_M/TILE_HEIGHT)*(SIZE_N/TILE_WIDTH); i++)
        {
            dC_spilled_row_cnt_offset[i] = tmp_row_cnt_size;
            dC_spilled_nnz_offset[i] = tmp_nnz_size;
            tmp_row_cnt_size += dC_spilled_row_cnt[i];
            tmp_nnz_size += dC_spilled_nnz[i];
        }
        *dC_spilled_row_buffersize = tmp_row_cnt_size;
        *dC_spilled_nnz_buffersize = tmp_nnz_size;
    }
}

template <typename int32_or_64>
__global__ void spgemm_compute_spilled(int *dC_spilled_row_row_idx,
                                       int *dC_spilled_row_tile_idx,
                                       int *dA_tiled_csr_offset_gmem,
                                       int *dA_tiled_csr_column_gmem,
                                       float *dA_tiled_csr_value_gmem,
                                       int *dA_tile_nnz_acc,
                                       int *dB_group_id_gmem,
                                       int32_or_64 *MatB_bit,
                                       float *dB_group_ele_row_val,
                                       int *dB_spilled_row_hash_table_reverse,
                                       int *dB_tile_spilled_csrRowPtr,
                                       int *dB_tile_spilled_csrColInd,
                                       float *dB_tile_spilled_csrVal,
                                       int *dB_spilled_row_cnt_offset,
                                       int *dB_spilled_nnz_offset,
                                       int *dC_spilled_csr_column,
                                       float *dC_spilled_csr_value
                                        )
{
    float result[TILE_WIDTH];
    // int bid = blockIdx.x + blockIdx.y * gridDim.x;
    // int tid = bid * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int rowC_ind = dC_spilled_row_row_idx[tid];
    int tileC_id = dC_spilled_row_tile_idx[tid];
    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        int tileA_id = SIZE_K/SPLIT_K * tileC_id/(SIZE_N/TILE_WIDTH) + k;
        int tileB_id = k * (SIZE_N/TILE_WIDTH) + tileC_id%(SIZE_N/TILE_WIDTH);
        int tile_nnz_acc = dA_tile_nnz_acc[tileA_id];

        int csr_offset_start = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+rowC_ind];
        int csr_offset_end = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+rowC_ind+1];

        for (int i = csr_offset_start; i < csr_offset_end; i++)
        {
            int entry_col = dA_tiled_csr_column_gmem[tile_nnz_acc + i];
            float value_A = dA_tiled_csr_value_gmem[tile_nnz_acc + i];
            int entry = (k * SPLIT_K + entry_col) * (SIZE_N/TILE_WIDTH)+tileC_id%(SIZE_N/TILE_WIDTH);
            int row_group_id = dB_group_id_gmem[entry];
            if (row_group_id != -1)
            {
                for (int i = 0; i < TILE_WIDTH; i++)
                {
                    if (MatB_bit[entry] >> i & 0x01)
                    {
                        result[i] += value_A * dB_group_ele_row_val[(MAX_GROUP_NUM * tileB_id + row_group_id) * TILE_WIDTH + i];
                    }
                }
            }
            else
            {
                int row_in_csr = dB_spilled_row_hash_table_reverse[entry_col + tileB_id * SPLIT_K];
                int start_offset;
                if (row_in_csr == 0)
                {
                    start_offset = 0;
                }
                else 
                {
                    start_offset = dB_tile_spilled_csrRowPtr[dB_spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                }
                for (int j = start_offset; j < dB_tile_spilled_csrRowPtr[dB_spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                {
                    int col_ind = dB_tile_spilled_csrColInd[dB_spilled_nnz_offset[tileB_id] + j];
                    result[col_ind] += value_A * dB_tile_spilled_csrVal[dB_spilled_nnz_offset[tileB_id] + j];
                }
            }
        }
    }
    for (int i = 0; i < TILE_WIDTH; i++)
    {
        if (result[i] != 0.0f)
        {
            dC_spilled_csr_value[tid] = result[i];
            dC_spilled_csr_column[tid] = i;
        }
    }
    
}

template <typename int32_or_64>
__global__ void spgemm_compute_1dthread(
                int32_or_64 *MatB_bit,
                float *dA_dense_gmem,
                float *dB_dense_gmem,
                int *group_id_gmem,
                int *spilled_row_hash_table_reverse_gmem,
                float *dB_group_ele_row_val,
                int *dB_spilled_row_cnt_offset,
                int *dB_spilled_nnz_offset,
                float *dB_tile_spilled_csrVal,
                int *dB_tile_spilled_csrColInd,
                int *dB_tile_spilled_csrRowPtr,
                int *dA_tiled_csr_offset_gmem,
                int *dA_tiled_csr_column_gmem,
                float *dA_tiled_csr_value_gmem,
                int *dA_tile_nnz_acc,
                int *dA_tile_nnz,
                int *dC_output_group_idx,
                float *final_result_gmem
            )
{
    // input buffers
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];
    __shared__ int32_or_64 MatB_bit_smem[SPLIT_K];
    // __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    // __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];
    // __shared__ float tiled_csr_value_smem[MAX_TILEA_NNZ];
    __shared__ float group[TILE_WIDTH][MAX_GROUP_NUM];

    // intermediate buffers
    // __shared__ int32_or_64 group_indicator[TILE_HEIGHT][BIT_WIDTH][MAX_GROUP_NUM];
    // __shared__ float result[TILE_HEIGHT][BIT_WIDTH][TILE_WIDTH];
    // float result[TILE_WIDTH];
    // int32_or_64 group_indicator[BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ int32_or_64 group_indicator[OUTPUT_MAX_GROUP_NUM][BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ float result[OUTPUT_MAX_GROUP_NUM][BIT_WIDTH][TILE_WIDTH];

    // for (int i = 0; i < MAX_TILEA_NNZ; i++)
    // {
    //     tiled_csr_column_smem[i] = 0;
    //     tiled_csr_value_smem[i] = 0;
    // }

    int row_group_id;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int output_group_idx = dC_output_group_idx[bid * blockDim.x + threadIdx.x];
    // if (output_group_idx == -1)
    // {
    //     return;
    // }
    // int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    // int group[MAX_GROUP_NUM];

    // Declare the fragments
    nvcuda::wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> A_frag;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> B_frag[8];
    nvcuda::wmma::fragment<wmma::accumulator, 8, 32, 16, float> C_frag[8];

    // Initialize the output to zero
    for (int i = 0; i < 8; i++)
    {
        nvcuda::wmma::fill_fragment(C_frag[i], 0.0f);
    }

    // // Load the inputs
    // nvcuda::wmma::load_matrix_sync(a_frag, (half *)&group[0][0], 16);
    // nvcuda::wmma::load_matrix_sync(b_frag, (half *)&group[0][0], 16);

    int tmp;
    int rowB_ind;
    int entry;
    int col_ind, ind, entry_col;
    int tileA_id, tileB_id, tile_nnz_acc;

    int csr_column[MAX_LINE_NNZ_A];
    float csr_value[MAX_LINE_NNZ_A];
    int line_csr_offset_start, line_csr_offset_end;

    int B_frag_idx = threadIdx.x/32; // 256/32=8 output groups per wave
    int output_row_idx = 2 * threadIdx.x/32;
    int lane_id = threadIdx.x % 32;

    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {
        tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        tileB_id = k * gridDim.x + blockIdx.x;
        tile_nnz_acc = dA_tile_nnz_acc[tileA_id];
        // int tile_nnz = dA_tile_nnz[tileA_id];

        line_csr_offset_start = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+threadIdx.x];
        line_csr_offset_end = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+threadIdx.x+1];

        // // Load MatA's tiled-csr data into shared memory
        // tiled_csr_offset_smem[threadIdx.x] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+threadIdx.x];
        // if (threadIdx.x == 0)
        // {
        //     tiled_csr_offset_smem[TILE_HEIGHT] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+TILE_HEIGHT];
        // }

        for (int z = 0; z < line_csr_offset_end - line_csr_offset_start; z++)
        {
            csr_column[z] = dA_tiled_csr_column_gmem[tile_nnz_acc + line_csr_offset_start + z];
            csr_value[z] = dA_tiled_csr_value_gmem[tile_nnz_acc + line_csr_offset_start + z];
        }

        // for (int i = 0; i < MAX_TILEA_NNZ/blockDim.x; i++)
        // {
        //     tiled_csr_column_smem[threadIdx.x + i*blockDim.x] = dA_tiled_csr_column_gmem[tile_nnz_acc + threadIdx.x + i*blockDim.x];
        //     tiled_csr_value_smem[threadIdx.x + i*blockDim.x] = dA_tiled_csr_value_gmem[tile_nnz_acc + threadIdx.x + i*blockDim.x];
        // }

        // Load MatB's group data into shared memory
        group[threadIdx.x/MAX_GROUP_NUM][threadIdx.x%MAX_GROUP_NUM] 
            = dB_group_ele_row_val[(MAX_GROUP_NUM * tileB_id + threadIdx.x%MAX_GROUP_NUM) * TILE_WIDTH + threadIdx.x/MAX_GROUP_NUM];

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        rowB_ind = k * SPLIT_K + threadIdx.x;
        entry = rowB_ind * gridDim.x + blockIdx.x;

        // Load MatB's group information into shared memory
        group_id_smem[threadIdx.x] = group_id_gmem[entry];

        // Load MatB's bit mask data into shared memory
        MatB_bit_smem[threadIdx.x] = MatB_bit[entry];

        spilled_row_hash_table_reverse_smem[threadIdx.x] 
            = spilled_row_hash_table_reverse_gmem[threadIdx.x + tileB_id * SPLIT_K];

        __syncthreads();


        // int rowA_ind = blockIdx.y * blockDim.x + threadIdx.x;
        #pragma unroll
        for (int z = 0; z < MAX_LINE_NNZ_A; z++)
        {
            if (z == line_csr_offset_end - line_csr_offset_start) break;
            entry_col = csr_column[z];
            row_group_id = group_id_smem[entry_col];
            int32_or_64 MatB_bit_row = MatB_bit_smem[entry_col];
            if (MatB_bit_row == 0) continue;
            // printf("row_group_id: %d\n", row_group_id);
            if (row_group_id != -1 && output_group_idx != -1)
            {
                for (int b = 0; b < BIT_WIDTH; b++)
                {
                    tmp = (__float_as_int(csr_value[z]) >> b) & 1 == 0x01;
                    if ((b % 2 + entry_col % 2) % 2 == 0)
                    {
                        // for (int z = 0; z < TILE_WIDTH; z++)
                        // {
                        //     if ((MatB_bit_smem[entry_col] >> z) & 0x01)
                        //     {
                        //         result[output_group_idx][b][z] += group[z][row_group_id];
                        //     }
                        // }
                        group_indicator[output_group_idx][b][row_group_id] |= MatB_bit_row;
                    }
                }
            }
            else if (row_group_id == -1 && output_group_idx != -1)
            {
                int row_in_csr = spilled_row_hash_table_reverse_smem[entry_col];
                int start_offset;
                if (row_in_csr == 0)
                {
                    start_offset = 0;
                }
                else 
                {
                    start_offset = dB_tile_spilled_csrRowPtr[dB_spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                }
                for (int j = start_offset; j < dB_tile_spilled_csrRowPtr[dB_spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                {
                    int col_ind = dB_tile_spilled_csrColInd[dB_spilled_nnz_offset[tileB_id] + j];
                    result[output_group_idx][1][col_ind] += csr_value[z] * dB_tile_spilled_csrVal[dB_spilled_nnz_offset[tileB_id] + j];
                }
            }
            
        }

        __syncthreads();

        // #pragma unroll
        // for (int b = 0; b < BIT_WIDTH; b++)
        // {
        //     #pragma unroll
        //     for (int i = 0; i < MAX_GROUP_NUM; i++)
        //     {
        //         if (group_indicator[output_row_idx][0][0] >> (32 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[0] += group[0][i];
        //         }
        //         if (group_indicator[output_row_idx][1][0] >> (32 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[1] += group[0][i];
        //         }
        //         if (group_indicator[output_row_idx+1][0][0] >> (32 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[2] += group[0][i];
        //         }
        //         if (group_indicator[output_row_idx+1][1][0] >> (32 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[3] += group[0][i];
        //         }
        //         if (group_indicator[output_row_idx][0][0] >> (16 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[4] += group[16][i];
        //         }
        //         if (group_indicator[output_row_idx][1][0] >> (16 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[5] += group[16][i];
        //         }
        //         if (group_indicator[output_row_idx+1][0][0] >> (16 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[6] += group[16][i];
        //         }
        //         if (group_indicator[output_row_idx+1][1][0] >> (16 - (threadIdx.x % 4)) & 0x01)
        //         {
        //             B_frag[B_frag_idx].x[7] += group[16][i];
        //         }
        //     }
        // }


        #pragma unroll
        for (int k = 0; k < 16; k++)
        {
            #pragma unroll
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            {
                int row = lane_id % 4 * 2 + k%2 + k%8/4*8;
                int col = lane_id / 4 + (k/2)/4*2+(k%2);
                int index = (k%8)/4;
                if (group_indicator[output_row_idx + index][row % 8][i] >> (32 - col) & 0x01)
                {
                    B_frag[B_frag_idx].x[k] += group[col][i];
                }
            }
        }


        // #pragma unroll
        // for (int b = 0; b < BIT_WIDTH; b++)
        // {
        //     #pragma unroll
        //     for (int i = 0; i < MAX_GROUP_NUM; i++)
        //     {
        //         if (group_indicator[b][i] == 0)
        //         {
        //             continue;
        //         }
        //         #pragma unroll
        //         for (int z = 0; z < TILE_WIDTH; z++)
        //         {
        //             if (((group_indicator[b][i] >> z) & 0x01) == 1)
        //             {
        //                 // result[output_group_idx][b][z] += group[z][i];
        //                 C_frag[output_group_idx][b].x[z] += group[z][i];
        //                 // result[threadIdx.x][threadIdx.y][z] += group[z][i];
        //                 // result[threadIdx.x][threadIdx.y][z] += i;
        //             }
        //         }
        //     }
        // }
    }

    for (int i = 0; i < 8; i++)
    {
        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(C_frag[i], A_frag, B_frag[i], C_frag[i]);

        // Store the output
        nvcuda::wmma::store_matrix_sync(final_result_gmem, C_frag[i], 16, wmma::mem_row_major);
    }

    // // compute with cuda core
    // #pragma unroll
    // for (int b = 0; b < BIT_WIDTH; b++)
    // {
    //     #pragma unroll
    //     for (int z = 0; z < TILE_WIDTH; z++)
    //     {
    //         ind = (blockIdx.y * blockDim.x + threadIdx.x) * SIZE_N + blockIdx.x * TILE_WIDTH + z;
    //         final_result_gmem[ind] += result[output_group_idx][b][z] * float(b);
    //         // final_result_gmem[ind] += result[threadIdx.x][threadIdx.y][i] * float(threadIdx.y);
    //     }
    // }

}


template <typename BitMaskType,
          typename ValueType,
          typename OutputType>
__global__ void spgemm_compute_1dthread_tcore(
                BitMaskType *dB_bit_mask_gmem,
                int *dB_group_id_gmem,
                int *dB_spilled_row_hash_table_reverse_gmem,
                ValueType *dB_group_value,

                int *dB_spilled_row_cnt_offset,
                int *dB_spilled_nnz_offset,

                ValueType *dB_tile_spilled_csrVal,
                int *dB_tile_spilled_csrColInd,
                int *dB_tile_spilled_csrRowPtr,

                int *dA_tiled_csr_offset_gmem,
                int *dA_tiled_csr_column_gmem,
                ValueType *dA_tiled_csr_value_gmem,
                int *dA_tile_nnz_acc,

                int *dC_output_group_idx,
                OutputType *dC_group_value,
                half *multiplicand,
                ValueType *d_probe
            )
{
    // input buffers
    __shared__ int group_id_smem[SPLIT_K];
    __shared__ int spilled_row_hash_table_reverse_smem[SPLIT_K];
    __shared__ BitMaskType MatB_bit_smem[SPLIT_K];
    __shared__ ValueType group[TILE_WIDTH][MAX_GROUP_NUM];
    __shared__ int tiled_csr_offset_smem[TILE_HEIGHT+1];
    __shared__ int tiled_csr_column_smem[MAX_TILEA_NNZ];
    __shared__ ValueType tiled_csr_value_smem[MAX_TILEA_NNZ];

    // intermediate buffers
    __shared__ BitMaskType group_indicator[OUTPUT_MAX_GROUP_NUM][BIT_WIDTH][MAX_GROUP_NUM];
    __shared__ OutputType result[OUTPUT_MAX_GROUP_NUM][BIT_WIDTH][TILE_WIDTH];

    if (threadIdx.x < 128)
    {
        group[threadIdx.x/4][threadIdx.x%4] = 0;
        for (int i = 0; i < 4; i++)
        {
            group_indicator[threadIdx.x/8][threadIdx.x%8][i] = 0;
        }
    }

    int row_group_id;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int output_group_idx = dC_output_group_idx[bid * blockDim.x + threadIdx.x];

    // Declare the fragments
    nvcuda::wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> A_frag;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> B_frag[8];
    nvcuda::wmma::fragment<wmma::accumulator, 8, 32, 16, float> C_frag[8];

    for (int i = 0; i < OUTPUT_MAX_GROUP_NUM * SIZE_M * SIZE_N / TILE_HEIGHT; i++)
    {
        dC_group_value[i] = 0;
    }

    // Initialize the output to zero
    for (int i = 0; i < 8; i++)
    {
        nvcuda::wmma::fill_fragment(B_frag[i], 0);
        nvcuda::wmma::fill_fragment(C_frag[i], 0);
    }

    // Load the inputs
    nvcuda::wmma::load_matrix_sync(A_frag, multiplicand, 16);
    // nvcuda::wmma::load_matrix_sync(b_frag, (half *)&group[0][0], 16);

    int tmp;
    int rowB_ind;
    int entry;
    int col_ind, ind, entry_col;
    int tileA_id, tileB_id, tile_nnz_acc;

    // int csr_column[MAX_LINE_NNZ_A];
    // float csr_value[MAX_LINE_NNZ_A];
    // int line_csr_offset_start, line_csr_offset_end;

    int warp_id = threadIdx.x/32; // 256/32=8 output groups per wave
    int output_row_idx = 2 * warp_id;
    int lane_id = threadIdx.x % 32;

    for (int k = 0; k < SIZE_K/SPLIT_K; k++)
    {

        if (threadIdx.x < 128)
        {
            for (int i = 0; i < 4; i++)
            {
                group_indicator[threadIdx.x/8][threadIdx.x%8][i] = 0;
            }
        }

        __syncthreads();


        tileA_id = SIZE_K/SPLIT_K * blockIdx.y + k;
        tileB_id = k * gridDim.x + blockIdx.x;
        tile_nnz_acc = dA_tile_nnz_acc[tileA_id];

        // Load MatA's tiled-csr data into shared memory
        tiled_csr_offset_smem[threadIdx.x] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+threadIdx.x];
        if (threadIdx.x == 0)
        {
            tiled_csr_offset_smem[TILE_HEIGHT] = dA_tiled_csr_offset_gmem[tileA_id*(TILE_HEIGHT+1)+TILE_HEIGHT];
        }

        // __syncthreads();

        for (int i = 0; i < MAX_TILEA_NNZ/blockDim.x; i++)
        {
            tiled_csr_column_smem[threadIdx.x + i*blockDim.x] = dA_tiled_csr_column_gmem[tile_nnz_acc + threadIdx.x + i*blockDim.x];
            tiled_csr_value_smem[threadIdx.x + i*blockDim.x] = dA_tiled_csr_value_gmem[tile_nnz_acc + threadIdx.x + i*blockDim.x];
        }
        __syncthreads();

        // Load MatB's group data into shared memory
        if (threadIdx.x < MAX_GROUP_NUM * TILE_WIDTH)
        {
            group[threadIdx.x/MAX_GROUP_NUM][threadIdx.x%MAX_GROUP_NUM] 
                = dB_group_value[(MAX_GROUP_NUM * tileB_id + threadIdx.x%MAX_GROUP_NUM) * TILE_WIDTH + threadIdx.x/MAX_GROUP_NUM];
        }

        __syncthreads();

        // SPLIT_K/blockDim.x = 256/32 = 8 = blockDim.y
        rowB_ind = k * SPLIT_K + threadIdx.x;
        entry = rowB_ind * gridDim.x + blockIdx.x;

        // Load MatB's group information into shared memory
        group_id_smem[threadIdx.x] = dB_group_id_gmem[entry];

        // Load MatB's bit mask data into shared memory
        MatB_bit_smem[threadIdx.x] = dB_bit_mask_gmem[entry];

        spilled_row_hash_table_reverse_smem[threadIdx.x] 
            = dB_spilled_row_hash_table_reverse_gmem[threadIdx.x + tileB_id * SPLIT_K];

        __syncthreads();

        // signed char tmp = tiled_csr_value_smem[0] >> 1;
        if (bid == 0 && threadIdx.x == 0 && k == 0)
        {
            printf("smem_value: %d\n", tiled_csr_value_smem[0]);
            printf("smem_column: %d\n", tiled_csr_column_smem[0]);
            // printf("row_group_id: %d\n", group_id_smem[tiled_csr_column_smem[1]]);
            // printf("MatB_bit_smem: %d\n", MatB_bit_smem[tiled_csr_column_smem[1]]);
            printf("smem_offset: %d\n", tiled_csr_offset_smem[1]);
            printf("gmem_value: %d\n", dA_tiled_csr_value_gmem[0]);
            printf("tile_nnz_acc_0: %d\n", dA_tile_nnz_acc[tileA_id]);
            printf("output_group_idx: %d\n", output_group_idx);
        }

        for (int z = tiled_csr_offset_smem[threadIdx.x]; z < tiled_csr_offset_smem[threadIdx.x+1]; z++)
        {
            // if (z == line_csr_offset_end - line_csr_offset_start) break;
            entry_col = tiled_csr_column_smem[z];
            row_group_id = group_id_smem[entry_col];
            // if (bid == 0 && threadIdx.x == 2 && k == 2)
            // {
            //     printf("row_group_id: %d, z: %d\n", dB_group_id_gmem[(k * SPLIT_K + entry_col) * gridDim.x + blockIdx.x], z);
            // }
            BitMaskType MatB_bit_row = MatB_bit_smem[entry_col];
            if (MatB_bit_row == 0) continue;
            // printf("row_group_id: %d\n", row_group_id);
            if (row_group_id != -1 && output_group_idx != -1)
            {
                for (int b = 0; b < BIT_WIDTH; b++)
                {
                    // tmp = (__float_as_int(tiled_csr_value_smem[z]) >> b) & 1 == 0x01;
                    // if ((b % 2 + entry_col % 2) % 2 == 0)
                    if ((tiled_csr_value_smem[z] >> b) & 1)
                    {
                        // group_indicator[output_group_idx][b][row_group_id] |= MatB_bit_row;
                        atomicOr(&group_indicator[output_group_idx][b][row_group_id], MatB_bit_row);
                    }
                }
            }
            else if (row_group_id == -1 && output_group_idx != -1)
            {
                int row_in_csr = spilled_row_hash_table_reverse_smem[entry_col];
                int start_offset;
                if (row_in_csr == 0)
                {
                    start_offset = 0;
                }
                else 
                {
                    start_offset = dB_tile_spilled_csrRowPtr[dB_spilled_row_cnt_offset[tileB_id] + row_in_csr - 1];
                }
                for (int j = start_offset; j < dB_tile_spilled_csrRowPtr[dB_spilled_row_cnt_offset[tileB_id] + row_in_csr]; j++)
                {
                    int col_ind = dB_tile_spilled_csrColInd[dB_spilled_nnz_offset[tileB_id] + j];
                    result[output_group_idx][1][col_ind] += tiled_csr_value_smem[z] * dB_tile_spilled_csrVal[dB_spilled_nnz_offset[tileB_id] + j];
                }
            }
        }

        // if (threadIdx.x == 0)
        // {
        //     for (int i = 0; i < 256; i++)
        //     {
        //         for (int z = tiled_csr_offset_smem[i]; z < tiled_csr_offset_smem[i+1]; z++)
        //         {
        //             // if (z == line_csr_offset_end - line_csr_offset_start) break;
        //             entry_col = tiled_csr_column_smem[z];
        //             row_group_id = group_id_smem[entry_col];
        //             output_group_idx = dC_output_group_idx[bid * blockDim.x + i];
        //             BitMaskType MatB_bit_row = MatB_bit_smem[entry_col];
        //             if (MatB_bit_row == 0) continue;
        //             // printf("row_group_id: %d\n", row_group_id);
        //             if (row_group_id != -1 && output_group_idx != -1)
        //             {
        //                 for (int b = 0; b < BIT_WIDTH; b++)
        //                 {
        //                     // tmp = (__float_as_int(tiled_csr_value_smem[z]) >> b) & 1 == 0x01;
        //                     // if ((b % 2 + entry_col % 2) % 2 == 0)
        //                     if ((tiled_csr_value_smem[z] >> b) & 1)
        //                     {
        //                         // group_indicator[output_group_idx][b][row_group_id] |= MatB_bit_row;
        //                         atomicOr(&group_indicator[output_group_idx][b][row_group_id], MatB_bit_row);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        __syncthreads();

        if (bid == 0 && threadIdx.x == 0)
        {
            printf("group_indicator: %d, k: %d\n", group_indicator[0][0][0], k);
        }

        // if (bid == 0 && k == 0 && threadIdx.x == 0)
        // {
        //     for (int i = 0; i < 16*32; i++)
        //     {
        //         d_probe[i] = group_indicator[i/32][(i%32)/4][(i%32)%4];
        //     }
        // }


        #pragma unroll
        for (int m = 0; m < 16; m++)
        {
            #pragma unroll
            for (int i = 0; i < MAX_GROUP_NUM; i++)
            { 
                int row = lane_id % 4 * 2 + m % 2 + m % 8 / 4 * 8;
                // int col = lane_id / 4 + (m / 2) / 4 * 2 + (m % 2);
                int col = lane_id / 4 + 8 * ((m/8)*2+(m%4)/2);
                int index = (m % 8) / 4;
                if (group_indicator[output_row_idx + index][row % 8][i] >> (31 - col) & 0x01)
                {
                    B_frag[warp_id].x[m] += group[col][i];
                    // B_frag[warp_id].x[m] += 73;
                    // __threadfence();
                }
            }
        }

        __syncthreads();

        if (bid == 0 && threadIdx.x == 0)
        {
        for (int r = 0; r < 16; r++)
        {
            for (int s = 0; s < 8; s++)
            {
                for (int t = 0; t < 4; t++)
                {
                    for (int p = 0; p < 32; p++)
                    {
                        if (group_indicator[r][s][t] >> (31-p) & 0x01)
                        {
                            d_probe[(r*8 + s)*32+p] += group[p][t];
                        }
                    }
                }
            }
        }
        }
        // printf("Stop it. Get some help: %d\n", d_probe[258]);

        __syncthreads();

    }

    // if (bid == 0 && threadIdx.x == 0)
    // {
    // for (int i = 0; i < 16; i++)
    // {
    //     for (int b = 0; b < BIT_WIDTH; b++)
    //     {
    //         for (int z = 0; z < TILE_WIDTH; z++)
    //         {
    //             signed char tmp;
    //             if (b == 0) tmp = 1;
    //             if (b == 1) tmp = 2;
    //             if (b == 2) tmp = 4;
    //             if (b == 3) tmp = 8;
    //             if (b == 4) tmp = 16;
    //             if (b == 5) tmp = 32;
    //             if (b == 6) tmp = 64;
    //             if (b == 7) tmp = -128;
    //             dC_group_value[i*TILE_WIDTH+z] += d_probe[(i*8 + b)*TILE_WIDTH+z] * tmp;
    //         }
    //     }
    // }
    // }

    // for (int i = 0; i < 8; i++)
    // {
    //     nvcuda::wmma::fill_fragment(B_frag[i], i);
    // }

    if (bid == 0)
    {
        for (int i = 0; i < 1; i++)
        {
            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(C_frag[i], A_frag, B_frag[i], C_frag[i]);

            // Store the output
            // nvcuda::wmma::store_matrix_sync(dC_group_value, C_frag[i], 32, wmma::mem_row_major);
            nvcuda::wmma::store_matrix_sync(dC_group_value + bid*OUTPUT_MAX_GROUP_NUM*TILE_WIDTH + 8*32*i, C_frag[i], 32, wmma::mem_row_major);

            // __syncthreads();
        }
    }


}


#endif