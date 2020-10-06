#include "transposer.hh"

void transposer::cuda::sort(int INPUT_ARRAY input, int * output, int len) {

    transposer::cuda::segsort(input, output, len);

    // alloco spazio necessario
    int* buffer[2] = { output, utils::cuda::allocate<int>(len) };
    int full = 0;

    // applico merge
    for(int BLOCK_SIZE = SEGSORT_ELEMENTS_PER_BLOCK; BLOCK_SIZE < len; BLOCK_SIZE *= 2) {
        
        DPRINT_MSG("Block size=%d", BLOCK_SIZE)
        segmerge_sm_step(buffer[full], buffer[1-full], len, BLOCK_SIZE);
        
        full = 1 - full;
        DPRINT_ARR_CUDA_BLOCK(buffer[full], len, BLOCK_SIZE*2)
    }

    // eventualmente copio nell'array di output nel caso non sia stato l'ultimo
    // ad essere riempito...
    if(full != 0) {
        utils::cuda::copy(output, buffer[1], len);
    }

    // dealloco array temporaneo;
    utils::cuda::deallocate(buffer[1]);
}

void transposer::reference::sort(int INPUT_ARRAY input, int * output, int len) {

    utils::copy_array(output, input, len);
    std::sort(output, output + len);
}

bool transposer::component_test::sort() {

    const int N = 10;
    // input
    int *arr = utils::random::generate_array(1, 100, N);
    DPRINT_ARR(arr, N)

    // reference implementation
    int *sort_arr = new int[N];
    transposer::reference::sort(arr, sort_arr, N);

    // cuda implementation
    int *sort_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
    int *sort_cuda_out = utils::cuda::allocate<int>(N);
    transposer::cuda::sort(sort_cuda_in, sort_cuda_out, N);
    int *sort_arr_2 = new int[N]; 
    utils::cuda::recv(sort_arr_2, sort_cuda_out, N);

    DPRINT_ARR(sort_arr, N)
    DPRINT_ARR(sort_arr_2, N)
    bool ok = utils::equals<int>(sort_arr, sort_arr_2, N);

    utils::cuda::deallocate(sort_cuda_in);
    utils::cuda::deallocate(sort_cuda_out);
    delete arr, sort_arr, sort_arr_2;
    
    return ok;
}