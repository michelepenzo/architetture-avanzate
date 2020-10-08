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

/*void procedures::cuda::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len) {
    
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
*/



__global__ 
void parallel_histogram_kernel(int INPUT_ARRAY idx, int idx_len, int * inter, int * intra, int HISTO_ROW_LEN) {
    
    int j = blockIdx.x;
    
    // allineo inter sulla porzione di array che sto usando
    inter += HISTO_ROW_LEN * (j+1);

    // costanti che servono per allineare intra sulla porzione di array che sto usando
    const int BLOCK_SIZE = DIV_THEN_CEIL(idx_len, HISTOGRAM_BLOCKS);
    const int START = BLOCK_SIZE * j;
    const int END = min(BLOCK_SIZE * (j+1), idx_len);

    // ogni [blocco di thread] segna i contributi di una [porzione di idx]
    for(int i = START; i < END; i++) {
        int index = idx[i];
        intra[i] = inter[index];
        inter[index]++;
    }
}

__global__
void vertical_scan_kernel(int * inter, int * ptr, int HISTO_ROW_LEN) {

    int j = blockIdx.x;

    // vertical scan di ogni colonna
    int i = 0;
    for(i = 0; i < HISTOGRAM_BLOCKS; i++) {
        inter[HISTO_ROW_LEN * (i+1) + j] += inter[HISTO_ROW_LEN * i + j];
    }
    ptr[j] = inter[HISTO_ROW_LEN*i + j];
}

void procedures::cuda::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * inter, int * intra, int * ptr, int ptr_len) {

    parallel_histogram_kernel<<<HISTOGRAM_BLOCKS, 1>>>(idx, idx_len, inter, intra, ptr_len);
    CUDA_CHECK_ERROR

    vertical_scan_kernel<<<ptr_len, 1>>>(inter, ptr, ptr_len);
    CUDA_CHECK_ERROR
}

void procedures::reference::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * inter, int * intra, int * ptr, int ptr_len) {
    
    const int BLOCK_SIZE = DIV_THEN_CEIL(idx_len, HISTOGRAM_BLOCKS);
    const int HISTO_ROW_LEN = ptr_len;

    // parallel histogram
    for(int tid = 0; tid < HISTOGRAM_BLOCKS; tid++) {

        const int START_INTER = (tid + 1) * HISTO_ROW_LEN;

        const int START_INTRA = tid * BLOCK_SIZE;

        for(int i = 0; i < BLOCK_SIZE && START_INTRA + i < idx_len; i++) {
            int index = START_INTER + idx[START_INTRA + i];
            intra[START_INTRA + i] = inter[index];
            inter[index]++;
        }
    }

    // vertical scan (join histograms)
    for(int tid = 0; tid < HISTOGRAM_BLOCKS; tid++) {

        const int START_INTER_0 = tid * HISTO_ROW_LEN;
        const int START_INTER_1 = (tid + 1) * HISTO_ROW_LEN;

        // prefix scan verticale
        for(int i = 0; i < HISTO_ROW_LEN; i++) {
            inter[START_INTER_1 + i] += inter[START_INTER_0 + i];
        }
    }

    // copy last row of inter to pointer
    const int START_INTER_LAST = HISTOGRAM_BLOCKS * HISTO_ROW_LEN;
    for(int i = 0; i < HISTO_ROW_LEN; i++) {
        ptr[i] = inter[START_INTER_LAST + i];
    }
}
