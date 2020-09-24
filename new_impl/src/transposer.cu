#include "transposer.hh"

matrix::SparseMatrix* transposer::transpose(matrix::SparseMatrix *sm, Mode mode) {

    matrix::SparseMatrix* result = NULL;
    int esito = COMPUTATION_ERROR;

    if(mode == SERIAL) {
        result = new matrix::SparseMatrix(sm->n, sm->m, sm->nnz, matrix::ALL_ZEROS_INITIALIZATION);
        esito = reference::serial_csr2csc(
            sm->m, sm->n, sm->nnz, 
            sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
            result->csrRowPtr, result->csrColIdx, result->csrVal);
        
    } else if(mode == MERGE) {
        // result = new matrix::SparseMatrix(sm->n, sm->m, sm->nnz, matrix::ALL_ZEROS_INITIALIZATION);
        // esito = merge_host_csr2csc(
        //     11, sm->m, sm->n, sm->nnz, 
        //     sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
        //     result->csrRowPtr, result->csrColIdx, result->csrVal);
        // 
    } else {
        return NULL;
    }

    if(esito == COMPUTATION_ERROR) {
        if(result != NULL) { delete result; }
        return NULL;
    } else {
        return result;
    } 
}

// ===============================================================================
// INDEX TO POINTERS =============================================================
// ===============================================================================

#define HISTOGRAM_BLOCKS 2

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

void transposer::cuda::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len) {
    
    DPRINT_MSG("Begin allocation of array with len=%d", HISTOGRAM_BLOCKS * ptr_len)
    int* histogram_blocks = utils::cuda::allocate_zero<int>(HISTOGRAM_BLOCKS * ptr_len);

    DPRINT_MSG("Calling 'histogram_blocks_kernel' with grid=%d, blocks=%d", DIV_THEN_CEIL(idx_len, 1024), 1024)
    histogram_blocks_kernel<<<DIV_THEN_CEIL(idx_len, 1024), 1024>>>(idx, idx_len, histogram_blocks, ptr_len);
    DPRINT_ARR_CUDA(histogram_blocks, HISTOGRAM_BLOCKS * ptr_len)
    CUDA_CHECK_ERROR

    DPRINT_MSG("Calling 'histogram_merge_kernel' with grid=%d, blocks=%d, allocated shared=%d", ptr_len, 1, HISTOGRAM_BLOCKS)
    histogram_merge_kernel<<<DIV_THEN_CEIL(ptr_len, 1024), 1024>>>(histogram_blocks, ptr_len, ptr);
    CUDA_CHECK_ERROR

    utils::cuda::deallocate(histogram_blocks);
}

void transposer::reference::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len) {
    for(int i = 0; i < idx_len; i++) {
        ASSERT_LIMIT(idx[i]+1, ptr_len);
        ptr[idx[i]+1]++;
    }
}

// ===============================================================================
// POINTERS TO INDEX =============================================================
// ===============================================================================

__global__ 
void pointers_to_indexes_kernel(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    if(i < ptr_len) {
        int start = ptr[i], end = ptr[i+1];
        for(int j = start; j < end; j++) {
            idx[j] = i;
        }
    }
}

void transposer::cuda::pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    pointers_to_indexes_kernel<<<ptr_len, 1>>>(ptr, ptr_len, idx, idx_len);
    CUDA_CHECK_ERROR
}

void transposer::reference::pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len) {
    for(int j = 0; j < ptr_len; j++) {
        for(int i = ptr[j]; i < ptr[j+1]; i++) {
            ASSERT_LIMIT(i, idx_len);
            idx[i] = j;
        }
    }
}

// ===============================================================================
// SORT3 =========================================================================
// ===============================================================================

__global__ 
void merge3_kernel(
    int INPUT_ARRAY key_in, int INPUT_ARRAY val1_in, int INPUT_ARRAY val2_in, 
    int *key_out, int *val1_out, int *val2_out, int block_len, int full_len
) {

}


void transposer::cuda::sort3(
    int INPUT_ARRAY key_in, int INPUT_ARRAY val1_in, int INPUT_ARRAY val2_in, 
    int *key_out, int *val1_out, int *val2_out, int len
) {

}


// ===============================================================================
// SERIAL IMPLEMENTATION =========================================================
// ===============================================================================

int transposer::reference::serial_csr2csc(
    int m, int n, int nnz, 
    int INPUT_ARRAY csrRowPtr, int INPUT_ARRAY csrColIdx, float INPUT_ARRAY csrVal, 
    int *cscColPtr, int *cscRowIdx, float *cscVal
) {
    // 1. costruisco `cscColPtr` come istogramma delle frequenze degli elementi per ogni colonna
    DPRINT_MSG("Step 1: idx to ptr")
    indexes_to_pointers(csrColIdx, nnz, cscColPtr, n+1);

    // 2. applico prefix_sum per costruire corretto `cscColPtr` (ogni cella tiene conto dei precedenti)
    DPRINT_MSG("Step 2: prefix sum")
    utils::prefix_sum(cscColPtr, n+1);

    // 3. sistemo indici di riga e valori
    DPRINT_MSG("Step 3: fix row, value arrays")
    int* curr = new int[n](); 

    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            int col = csrColIdx[j];
            int loc = cscColPtr[col] + curr[col];
            curr[col]++;
            cscRowIdx[loc] = i;
            cscVal[loc] = csrVal[j];
        }
    }

    DPRINT_MSG("End")

    delete[] curr;
    return COMPUTATION_OK;
}


// ===============================================================================
// COMPONENT TEST ================================================================
// ===============================================================================

bool transposer::component_test::indexes_to_pointers() {

    const int N = 10000, NNZ = 10000;
    // input
    int *idx = utils::random::generate_array(0, N-1, NNZ);
    DPRINT_ARR(idx, NNZ)

    // reference implementation
    int *ptr = new int[N+1];
    transposer::reference::indexes_to_pointers(idx, NNZ, ptr, N+1);
    DPRINT_ARR(ptr, N+1)

    // cuda implementation
    int *idx_cuda = utils::cuda::allocate_send<int>(idx, NNZ);
    int *ptr_cuda = utils::cuda::allocate_zero<int>(N+1);
    transposer::cuda::indexes_to_pointers(idx_cuda, NNZ, ptr_cuda, N+1);
    int *ptr2 = new int[N+1]; utils::cuda::recv(ptr2, ptr_cuda, N+1);
    DPRINT_ARR(ptr2, N+1)

    bool ok = utils::equals<int>(ptr, ptr2, N+1);

    utils::cuda::deallocate(idx_cuda);
    utils::cuda::deallocate(ptr_cuda);
    delete idx, ptr, ptr2;
    
    return ok;
} 

bool transposer::component_test::pointers_to_indexes() {

    const int N = 10000, NNZ = 10000;

    int *ptr = utils::random::generate_array(0, 1, N+1);
    ptr[N] = 0;
    utils::prefix_sum(ptr, N+1);
    DPRINT_ARR(ptr, N+1)

    // reference implementation
    int *idx = new int[NNZ];
    reference::pointers_to_indexes(ptr, N+1, idx, NNZ);
    DPRINT_ARR(idx, NNZ)

    // cuda implementation
    int *ptr_cuda = utils::cuda::allocate_send<int>(ptr, N+1);
    int *idx_cuda = utils::cuda::allocate_zero<int>(NNZ);
    transposer::cuda::pointers_to_indexes(ptr_cuda, N+1, idx_cuda, NNZ);
    int *idx2 = new int[N+1]; utils::cuda::recv(idx2, idx_cuda, NNZ);
    DPRINT_ARR(idx2, NNZ)

    bool ok = utils::equals<int>(idx, idx2, NNZ);

    utils::cuda::deallocate(idx_cuda);
    utils::cuda::deallocate(ptr_cuda);
    delete ptr, idx, idx2;
    
    return ok;
}
