#include "procedures.hh"

__global__ 
void pointers_to_indexes_kernel(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    int i = blockIdx.x;
    int b = blockDim.x;

    if(i < ptr_len) {
        int start = ptr[i], end = ptr[i+1];
        for(int j = start + threadIdx.x; j < end; j += b) {
            idx[j] = i;
        }
    }
}

void procedures::cuda::pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    pointers_to_indexes_kernel<<<ptr_len, 1024>>>(ptr, ptr_len, idx, idx_len);
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
