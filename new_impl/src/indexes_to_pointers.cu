#include "procedures.hh"

// ===============================================================================
// INDEX TO POINTERS =============================================================
// ===============================================================================

__global__ 
void histogram_blocks_kernel(int INPUT_ARRAY elements, int n_elem, int * histogram_blocks, int hist_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    
    if(i < n_elem) {
        int j = blockIdx.x % HISTOGRAM_BLOCKS;
        // each of the HISTOGRAM_BLOCKS grid works on a different partial histogram
        int * this_histogram_block = histogram_blocks + j * hist_len;
        // needs to be atomic: two thread may (unlikely) access the same element at the same time
        // [alternative: one single thread per block (inefficient?)]
        atomicAdd(this_histogram_block + elements[i] + 1, 1);
    }
}

__global__ 
void histogram_merge_kernel(int INPUT_ARRAY histogram_blocks, int hist_len, int * histogram) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id

    if(i < hist_len) {
        int sum = 0;
        for(int k = 0; k < HISTOGRAM_BLOCKS; k++) {
            int * this_histogram_block = histogram_blocks + k * hist_len;
            sum += this_histogram_block[i];
        }
        histogram[i] = sum;
    }
}

void procedures::cuda::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len) {
    
    DPRINT_MSG("Begin allocation of array with len=%d", HISTOGRAM_BLOCKS * ptr_len)
    int* histogram_blocks = utils::cuda::allocate_zero<int>(HISTOGRAM_BLOCKS * ptr_len);

    DPRINT_MSG("Calling 'histogram_blocks_kernel' with grid=%d, blocks=%d", DIV_THEN_CEIL(idx_len, 1024), 1024)
    histogram_blocks_kernel<<<DIV_THEN_CEIL(idx_len, 1024), 1024>>>(idx, idx_len, histogram_blocks, ptr_len);
    CUDA_CHECK_ERROR
    DPRINT_ARR_CUDA(histogram_blocks, HISTOGRAM_BLOCKS * ptr_len)

    DPRINT_MSG("Calling 'histogram_merge_kernel' with grid=%d, blocks=%d, allocated shared=%d", ptr_len, 1, HISTOGRAM_BLOCKS)
    histogram_merge_kernel<<<DIV_THEN_CEIL(ptr_len, 1024), 1024>>>(histogram_blocks, ptr_len, ptr);
    CUDA_CHECK_ERROR

    utils::cuda::deallocate(histogram_blocks);
}

void procedures::reference::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len) {
    for(int i = 0; i < idx_len; i++) {
        ASSERT_LIMIT(idx[i]+1, ptr_len);
        ptr[idx[i]+1]++;
    }
}
