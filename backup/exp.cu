#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

#include "common.h"

__global__ void generate_groups()
{
    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int tid = bid * blockDim.x + threadIdx.x;
    __shared__ row_group[MAX_GROUP_NUM];

    if (tid > 32){
        return;
    }

    int and_result = atomicAnd(row_group[group_idx], mask_row[tid]);
    while (and_result != 0) {
        group_idx++;
        and_result = atomicAnd(row_group[group_idx], mask_row[tid]);
    }

    int expected = row_group[group_idx];
    int or_result = atomicOr(row_group[group_idx], mask_row[tid]);
    int old_value = atomicCAS(row_group[group_idx], expected, or_result);

    while (expected != old_value) {
        group_idx++;
        and_result = atomicAnd(row_group[group_idx], mask_row[tid]);
        while (and_result != 0) {
            group_idx++;
            and_result = atomicAnd(row_group[group_idx], mask_row[tid]);
        }
        expected = row_group[group_idx];
        or_result = atomicOr(row_group[group_idx], mask_row[tid]);
        old_value = atomicCAS(row_group[group_idx], expected, or_result);
    }
    for (int i = 0; i < 64; i++) {
        if (old_value << i == 1) {
            group_ele_row_idx[group_idx][i] = tid;
        }
    }
    group_mask[group_idx] = old_value;
    group_id[tid] = group_idx;
}

__global__ void ld_rows_to_groups()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // column per thread
    // Load rows to groups and // add up groups into registers
    for (int i = 0; i < MAX_GROUP_NUM; i++) {
        // Need modification, should read from the CSR or other condensed formats.
        group[i][tid] = row[group_ele_row_idx[i][tid]][tid];
    }
    
    // MAX_GROUP_NUM = 4
    group[4][tid] = group[0][tid] + group[1][tid];
    group[5][tid] = group[0][tid] + group[2][tid];
    group[6][tid] = group[0][tid] + group[3][tid];
    group[7][tid] = group[1][tid] + group[2][tid];
    group[8][tid] = group[1][tid] + group[3][tid];
    group[9][tid] = group[2][tid] + group[3][tid];
    group[10][tid] = group[4][tid] + group[2][tid];
    group[11][tid] = group[4][tid] + group[3][tid];
    group[12][tid] = group[5][tid] + group[3][tid];
    group[13][tid] = group[7][tid] + group[3][tid];
    group[14][tid] = group[10][tid] + group[3][tid];
    group[15][tid] = group[4][tid] + group[9][tid];

    // row per thread
    // Generate group indicator for each row in Matrix A
    for (int i = 0; i < nnzlen[tid]; i++) {
        group_indicator[tid][group_id[csr_row[tid][i]]] 
            = atomicAnd(group_indicator[tid][group_id[csr_row[tid][i]]], mask_row[csr_row[tid][i]]);
    }

    // column per thread
    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < MAX_GROUP_NUM; j++) {
            register_idx = atomicAnd(register_idx, group_indicator[i][j][tid] << j);
        }
        result[i][tid] = group[register_idx][tid];
    }

}


__global__ void layout_transform()
{
    int bid = blockIdx.y * gridDim.x + blockIdx.x;  
    int tid = bid * blockDim.x + threadIdx.x;

    int assigned_row_ind = blockIdx.y * m + threadIdx.x / m;
    int assigned_col_ind = blockIdx.x * k;
    int assigned_bit_pos = threadIdx.x % BIT_WIDTH;

    __shared__ int64_t tiled_mask_rowB[k];

    // Transform the CSR of MatrixA to a tiled form.
    if (threadIdx.x == 0) 
    {
        for (int i = csrRowPtrA[assigned_row_ind]; i < csrRowPtrA[assigned_row_ind+1]; i++)
        {
            if (csrColIndA[i] > k * blockIdx.x && csrColIndA[i] < k * (blockIdx.x+1))
            {
                tiled_csrColInd[].append(csrColIndA[i] - k * blockIdx.x);
                tiled_csrVal[].append(csrVal[i]);
            }
        }
    }

    for (int i = 0; i < sizeof(tiled_csrColInd); i++)
    {
        for (int b = 0; b < BIT_WIDTH; b++)
        {
            if(__float_as_int(tiled_csrVal[i]) << b == 1)
            {
                row_ind = tiled_csrColInd[i];
                group_indicator[tid][b][group_id[row_ind]] 
                    = atomicAnd(group_indicator[tid][b][group_id[row_ind]], tiled_mask_rowB[row_ind]);
            }
        }
    }

    // column per thread
    for (int i = 0; i < row_count; i++) 
    {
        for (int b = 0; b < BIT_WIDTH; b++)
        {
            for (int j = 0; j < MAX_GROUP_NUM; j++) 
            {
                register_idx = atomicAnd(register_idx, group_indicator[i][b][j][tid] << j); // this is one bit
            }
            result[i][b][tid] = group[register_idx][tid];
        }
    }



}

__global__ void merge_rows_in_advance()
{

}

__global__ void pre_merge_rows_by_group()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < nnzrow; i++){
        add_list[tid].append = group_id[csr_row[i]];
    }

}

int main() 
{
    
}