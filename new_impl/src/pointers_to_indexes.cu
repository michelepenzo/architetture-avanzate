#include "procedures.hh"

__global__ 
void pointers_to_indexes_kernel(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    if(i < ptr_len) {
        int start = ptr[i], end = ptr[i+1];
        //printf("(%2d): start=ptr[%2d]=%2d, end=ptr[%2d]=%2d\n", i, i, start, i+1, end);
        for(int j = start; j < end; j++) {
            idx[j] = i;
        }
    }
}

void procedures::cuda::pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    pointers_to_indexes_kernel<<<ptr_len, 1>>>(ptr, ptr_len, idx, idx_len);
    CUDA_CHECK_ERROR
}

void procedures::reference::pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    for(int j = 0; j < ptr_len; j++) {
        for(int i = ptr[j]; i < ptr[j+1]; i++) {
            ASSERT_LIMIT(i, idx_len);
            idx[i] = j;
        }
    }
}
