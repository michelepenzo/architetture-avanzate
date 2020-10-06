#include "transposer.hh"

__device__
int find_position_in_unsorted_array(int INPUT_ARRAY array, int element_index, int len) {

    int destination_index = 0;

    // conta posizioni prima dell'elemento corrente (duplicati compresi)
    for(int i = 0; i < element_index; i++) {
        if(array[i] <= array[element_index]) {
            destination_index ++;
        }
    }

    // conta posizioni dopo l'elemento corrente (duplicati esclusi)
    for(int i = element_index + 1; i < len; i++) {
        if(array[i] < array[element_index]) {
            destination_index ++;
        }
    }

    return destination_index;
}


__global__
void segsort_kernel(int INPUT_ARRAY input, int * output, int len) {

    __shared__ int temp[SEGSORT_ELEMENTS_PER_BLOCK];
    int i = threadIdx.x;
    int j = blockIdx.x * SEGSORT_ELEMENTS_PER_BLOCK + threadIdx.x;

    // caricamento dati in shared memory
    temp[i] = (j < len) ? input[j] : INT32_MAX;
    __syncthreads();

    /// trovo la posizione del i-esimo elemento
    int index = find_position_in_unsorted_array(temp, i, SEGSORT_ELEMENTS_PER_BLOCK);
    __syncthreads();

    // porto alla posizione corretta
    temp[index] = temp[i];
    __syncthreads();

    // scaricamento dati in shared memory
    if(i < len) {
        output[j] = temp[i];
    }
}

void transposer::cuda::segsort(int INPUT_ARRAY input, int * output, int len) {

    const int SEGMENT_NUMBER = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    segsort_kernel<<<SEGMENT_NUMBER, SEGSORT_ELEMENTS_PER_BLOCK >>>(input, output, len);
    CUDA_CHECK_ERROR
}

void transposer::reference::segsort(int INPUT_ARRAY input, int * output, int len) {

    utils::copy_array(output, input, len);

    const int SEGMENT_NUMBER = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    for(int i = 0; i < SEGMENT_NUMBER; i++) {
        const int start = i * SEGSORT_ELEMENTS_PER_BLOCK;
        const int end = std::min((i + 1) * SEGSORT_ELEMENTS_PER_BLOCK, len);
        std::sort(output + start, output + end);
    }
}

bool transposer::component_test::segsort() {

    const int N = 10000000;
    // input
    int *arr = utils::random::generate_array(1, 100, N);
    DPRINT_ARR(arr, N)

    // reference implementation
    int *segsort_arr = new int[N];
    transposer::reference::segsort(arr, segsort_arr, N);
    DPRINT_ARR(segsort_arr, N)

    // cuda implementation
    int *segsort_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
    int *segsort_cuda_out = utils::cuda::allocate<int>(N);
    transposer::cuda::segsort(segsort_cuda_in, segsort_cuda_out, N);
    int *segsort_arr_2 = new int[N]; 
    utils::cuda::recv(segsort_arr_2, segsort_cuda_out, N);
    DPRINT_ARR(segsort_arr_2, N)

    bool ok = utils::equals<int>(segsort_arr, segsort_arr_2, N);

    utils::cuda::deallocate(segsort_cuda_in);
    utils::cuda::deallocate(segsort_cuda_out);
    delete arr, segsort_arr, segsort_arr_2;
    
    return ok;
}