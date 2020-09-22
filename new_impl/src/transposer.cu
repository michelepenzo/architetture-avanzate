#include "transposer.hh"

bool transposer::test::test_indexes_to_pointers() {
    return true;
}

/*

__global__ 
void indexes_to_pointers_kernel(int INPUT_ARRAY idx, int idx_len, int * ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    if(i < idx_len) {
        int start = idx[i], end = idx[i+1];
        for(int j = start; j < end; j++) {
            ptr[j] = i;
        }
    }
}

int CUDA_DIVIDE_IN_BLOCKS = 1;
int CUDA_BLOCK_SIZE = 1024;

void transposer::cuda::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len) {
    
    if(CUDA_DIVIDE_IN_BLOCKS == 1) {
        const int BLOCK_NUMBER = DIV_THEN_CEIL(idx_len, CUDA_BLOCK_SIZE);
        indexes_to_pointers_kernel<<<BLOCK_NUMBER, CUDA_BLOCK_SIZE>>>(idx, idx_len, ptr);
    } else {
        indexes_to_pointers_kernel<<<idx_len, 1>>>(idx, idx_len, ptr);
    }
    CUDA_CHECK_ERROR
}


bool transposer::test::test_indexes_to_pointers() {

    const int N = 10000, NNZ = 10000;
    int * idx = utils::create_random_array(0, N, NNZ);
    int * ptr1 = new int[N+1];
    int * ptr2 = new int[N+1];

    transposer::indexes_to_pointers(idx, NNZ, ptr1, N+1);

    int * idx_cuda;
    int * ptr_cuda;
    cudaMalloc(&idx_cuda, NNZ * sizeof(int));
    cudaMalloc(&ptr_cuda, (N+1) * sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpy(idx_cuda, idx, NNZ*sizeof(int), cudaMemcpyHostToDevice));
    transposer::cuda::indexes_to_pointers(idx_cuda, NNZ, ptr_cuda, N+1);
    CUDA_SAFE_CALL(cudaMemcpy(ptr2, ptr_cuda, (N+1)*sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(idx_cuda);
    cudaFree(ptr_cuda);

    bool ok = utils::equals(ptr1, ptr2, N+1);
    delete idx, ptr1, ptr2;
    return ok;
}   */