#include "procedures.hh"

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
    int start = blockIdx.x * SEGSORT_ELEMENTS_PER_BLOCK;
    int end = min((blockIdx.x+1) * SEGSORT_ELEMENTS_PER_BLOCK, len);

    if(i < end - start) {
        // caricamento dati in shared memory
        temp[i] = input[start + i];
        __syncthreads();

        /// trovo la posizione del i-esimo elemento
        int index = find_position_in_unsorted_array(temp, i, end - start);
        int element = temp[i];
        __syncthreads();

        // porto alla posizione corretta
        temp[index] = element;
        __syncthreads();

        // scaricamento dati in shared memory
        output[start + i] = temp[i];
        __syncthreads();
    }
    
}

__global__
void segsort3_kernel(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, float INPUT_ARRAY b_in, float * b_out) {

    __shared__ int temp[SEGSORT_ELEMENTS_PER_BLOCK];
    __shared__ int temp_a[SEGSORT_ELEMENTS_PER_BLOCK];
    __shared__ float temp_b[SEGSORT_ELEMENTS_PER_BLOCK];

    int i = threadIdx.x;
    int start = blockIdx.x * SEGSORT_ELEMENTS_PER_BLOCK;
    int end = min((blockIdx.x+1) * SEGSORT_ELEMENTS_PER_BLOCK, len);

    if(i < end - start) {
        // caricamento dati in shared memory
        temp[i]   = input[i + start];
        temp_a[i] =  a_in[i + start];
        temp_b[i] =  b_in[i + start];
        __syncthreads();

        /// trovo la posizione del i-esimo elemento
        int index = find_position_in_unsorted_array(temp, i, end - start);
        int element     = temp[i];
        int element_a   = temp_a[i];
        float element_b = temp_b[i];
        __syncthreads();

        // porto alla posizione corretta
        temp[index]   = element;
        temp_a[index] = element_a;
        temp_b[index] = element_b;
        __syncthreads();

        // scaricamento dati in shared memory
        output[start + i] = temp[i];
        a_out[start + i]  = temp_a[i];
        b_out[start + i]  = temp_b[i];
        __syncthreads();
    }
}

void procedures::cuda::segsort(int INPUT_ARRAY input, int * output, int len) {

    const int SEGMENT_NUMBER = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    segsort_kernel<<<SEGMENT_NUMBER, SEGSORT_ELEMENTS_PER_BLOCK >>>(input, output, len);
    CUDA_CHECK_ERROR
    //DPRINT_ARR_CUDA(input, len)
    //DPRINT_ARR_CUDA(output, len)
}

void procedures::cuda::segsort3(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, float INPUT_ARRAY b_in, float * b_out) {
    
    const int SEGMENT_NUMBER = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    segsort3_kernel<<<SEGMENT_NUMBER, SEGSORT_ELEMENTS_PER_BLOCK >>>(input, output, len, a_in, a_out, b_in, b_out);
    CUDA_CHECK_ERROR
}

void procedures::reference::segsort(int INPUT_ARRAY input, int * output, int len) {

    utils::copy_array(output, input, len);

    const int SEGMENT_NUMBER = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    for(int i = 0; i < SEGMENT_NUMBER; i++) {
        const int start = i * SEGSORT_ELEMENTS_PER_BLOCK;
        const int end = std::min((i + 1) * SEGSORT_ELEMENTS_PER_BLOCK, len);
        sort(input + start, output + start, end - start);
    }
}

void procedures::reference::segsort3(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, float INPUT_ARRAY b_in, float * b_out) {

    utils::copy_array(output, input, len);
    utils::copy_array(a_out,  a_in,  len);
    utils::copy_array(b_out,  b_in,  len);

    const int SEGMENT_NUMBER = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    for(int i = 0; i < SEGMENT_NUMBER; i++) {
        const int start = i * SEGSORT_ELEMENTS_PER_BLOCK;
        const int end = std::min((i + 1) * SEGSORT_ELEMENTS_PER_BLOCK, len);
        sort3(input + start, output + start, end - start, a_in + start, a_out + start, b_in + start, b_out + start);
    }
}