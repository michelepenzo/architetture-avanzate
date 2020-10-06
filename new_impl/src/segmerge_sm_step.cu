#include "transposer.hh"

__device__
void copy(int * output, int INPUT_ARRAY input, int len) {
    for(int i = 0; i < len; i++) {
        output[i] = input[i];
    }
}

__device__
int binary_search(int element_to_search, int INPUT_ARRAY input, int len) {

    int start = 0;
    int end = start + len;

    while(start < end) {
        int current = (start + end) / 2;
        if(input[current] < element_to_search) {
            start = current + 1;
        } else if(input[current] >= element_to_search) {
            end = current;
        }
    }

    return start;
}

__device__
int binary_search_last(int element_to_search, int INPUT_ARRAY input, int len) {

    int index = binary_search(element_to_search, input, len);
    while(input[index] == element_to_search && index < len) {
        index++;
    }
    return index;
}

__global__
void splitter_kernel(int INPUT_ARRAY input, int * splitter, int * indexA, int * indexB, int len, int BLOCK_SIZE) {

    int couple_block_id = blockIdx.x;

    // entrambi inputA, inputB esistono (eventualmente B ha lunghezza < BLOCK_SIZE se ultimo blocco)
    int * inputA = input + 2 * couple_block_id * BLOCK_SIZE;
    int * inputB = input + (2 * couple_block_id + 1) * BLOCK_SIZE;
    int lenA = BLOCK_SIZE;
    int endB = min((2 * couple_block_id + 2) * BLOCK_SIZE, len);
    int lenB = endB - (2 * couple_block_id + 1) * BLOCK_SIZE;

    // mi sposto verso gli indici corretti di splitter, indexA, indexB
    const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, SEGMERGE_SM_SPLITTER_DISTANCE);
    splitter = splitter + 2 * couple_block_id * SPLITTER_PER_BLOCKS;
    indexA = indexA + 2 * couple_block_id * SPLITTER_PER_BLOCKS;
    indexB = indexB + 2 * couple_block_id * SPLITTER_PER_BLOCKS;

    // riempio gli elementi
    int i;
    for(int i = 0; i < SPLITTER_PER_BLOCKS && i*SEGMERGE_SM_SPLITTER_DISTANCE < lenA; i++) {
        splitter[i] = inputA[i*SEGMERGE_SM_SPLITTER_DISTANCE];
        if(i > 0) {
            // shifto indietro di 1 per evitare il primo merge (di porzioni di array vuote)
            indexA[i - 1] = i*SEGMERGE_SM_SPLITTER_DISTANCE;
            indexB[i - 1] = binary_search(splitter[i], inputB, lenB);
        }
    }

    for(i = 0; i < SPLITTER_PER_BLOCKS && i*SEGMERGE_SM_SPLITTER_DISTANCE < lenB; i++) {
        splitter[SPLITTER_PER_BLOCKS + i] = inputB[i*SEGMERGE_SM_SPLITTER_DISTANCE];
        // shifto indietro di 1 per evitare il primo merge (di porzioni di array vuote)
        indexA[SPLITTER_PER_BLOCKS + i - 1] = binary_search_last(splitter[i], inputB, lenB);
        indexB[SPLITTER_PER_BLOCKS + i - 1] = i*SEGMERGE_SM_SPLITTER_DISTANCE;
    }

    // alla fine degli indici ci sono le dimensioni degli array
    indexA[SPLITTER_PER_BLOCKS + i - 1] = lenA;
    indexB[SPLITTER_PER_BLOCKS + i - 1] = lenB;
}

__global__
void uniform_merge_kernel(int INPUT_ARRAY input, int * output, int INPUT_ARRAY indexA, int INPUT_ARRAY indexB, int len, int BLOCK_SIZE) {

    __shared__ int temp_in[2 * SEGMERGE_SM_SPLITTER_DISTANCE];
    __shared__ int temp_out[2 * SEGMERGE_SM_SPLITTER_DISTANCE];

    // processa l'elemento dello splitter
    const int splitter_index = blockIdx.x;

    // recupera blocco sul quale stai lavorando
    const int SPLITTER_PER_BLOCK = DIV_THEN_CEIL(BLOCK_SIZE, SEGMERGE_SM_SPLITTER_DISTANCE);
    int couple_block_id = splitter_index / (2 * SPLITTER_PER_BLOCK);
    int item = splitter_index % (2 * SPLITTER_PER_BLOCK);

    // recupera estremi sui quali lavorare
    int * inputA = input + 2 * couple_block_id * BLOCK_SIZE;
    int * inputB = input + (2 * couple_block_id + 1) * BLOCK_SIZE;
    int startA = (item == 0) ? 0 : indexA[splitter_index-1];
    int startB = (item == 0) ? 0 : indexB[splitter_index-1];
    int endA   = indexA[splitter_index];
    int endB   = indexB[splitter_index];

    // carico gli elementi in temp_in
    copy(temp_in, inputA + startA, endA - startA);
    copy(temp_in + SEGMERGE_SM_SPLITTER_DISTANCE, inputB + startB, endB - startB);

    // effettuo merge
    for(int i = 0; i < endA - startA; i++) {
        int k = i + binary_search(temp_in[i], temp_in + SEGMERGE_SM_SPLITTER_DISTANCE, endB - startB);
        temp_out[k] = temp_in[i];
    }
    for(int i = 0; i < endB - startB; i++) {
        int element = temp_in[SEGMERGE_SM_SPLITTER_DISTANCE + i];
        int k = i + binary_search_last(element, temp_in, endA - startA);
        temp_out[k] = temp_in[i];
    }

    // salva output
    output = output + 2 * couple_block_id * BLOCK_SIZE;
    for(int i = 0; i < (endA - startA) + (endB - startB); i++) {
        output[startA + startB + i] = temp_out[i];
    }
}

void transposer::cuda::segmerge_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {
    
    // 1. lavoro su coppie di blocchi per estrarre gli splitter e gli indici necessari a lavorarci sopra
    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
    const int COUPLE_OF_BLOCKS = BLOCK_NUMBER / 2;
    const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, SEGMERGE_SM_SPLITTER_DISTANCE);
    const int SPLITTER_NUMBER = 2 * COUPLE_OF_BLOCKS * SPLITTER_PER_BLOCKS;

    int * splitter     = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * indexA       = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * indexB       = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * splitter_out = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * indexA_out   = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * indexB_out   = utils::cuda::allocate<int>(SPLITTER_NUMBER);

    splitter_kernel<<<COUPLE_OF_BLOCKS, 1>>>(input, splitter, indexA, indexB, len, BLOCK_SIZE);
    CUDA_CHECK_ERROR
    DPRINT_MSG("After splitter_kernel")
    DPRINT_ARR_CUDA(splitter, SPLITTER_NUMBER)
    DPRINT_ARR_CUDA(indexA, SPLITTER_NUMBER)
    DPRINT_ARR_CUDA(indexB, SPLITTER_NUMBER)

    // 2. riordino per blocchi l'array degli splitter e gli indici ad esso associati
    segmerge3_sm_step(splitter, splitter_out, SPLITTER_NUMBER, SPLITTER_PER_BLOCKS, indexA, indexA_out, indexB, indexB_out);
    DPRINT_MSG("After segmerge3_sm_step")
    DPRINT_ARR_CUDA(splitter_out, SPLITTER_NUMBER)
    DPRINT_ARR_CUDA(indexA_out, SPLITTER_NUMBER)
    DPRINT_ARR_CUDA(indexB_out, SPLITTER_NUMBER)

    // 3. eseguo il merge di porzioni di blocchi di dimensione uniforme
    uniform_merge_kernel<<<SPLITTER_NUMBER, 1>>>(input, output, indexA_out, indexB_out, len, BLOCK_SIZE);
    DPRINT_MSG("After uniform_merge_kernel")
    DPRINT_ARR_CUDA(input, len)
    DPRINT_ARR_CUDA(output, len)
    CUDA_CHECK_ERROR

    // 4. eventualmente copio il risultato dell' ultimo blocco di array rimasto spaiato
    if(BLOCK_NUMBER % 2 == 1) {
        utils::cuda::copy<int>(
            output + 2 * COUPLE_OF_BLOCKS * BLOCK_SIZE, 
            input +  2 * COUPLE_OF_BLOCKS * BLOCK_SIZE, 
            len - 2 * COUPLE_OF_BLOCKS * BLOCK_SIZE
        );
    }

    utils::cuda::deallocate(splitter);
    utils::cuda::deallocate(indexA);
    utils::cuda::deallocate(indexB);
    utils::cuda::deallocate(splitter_out);
    utils::cuda::deallocate(indexA_out);
    utils::cuda::deallocate(indexB_out);
}

void transposer::cuda::segmerge3_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {
    segmerge3_step(input, output, len, BLOCK_SIZE, a_in, a_out, b_in, b_out);
}

void transposer::reference::segmerge_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {
    segmerge_step(input, output, len, BLOCK_SIZE);
}

void transposer::reference::segmerge3_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {
    segmerge3_step(input, output, len, BLOCK_SIZE, a_in, a_out, b_in, b_out);
}


// ================================================
//  segmerge sm 
bool transposer::component_test::segmerge_sm_step() {

    const int N = 100;
    int BLOCK_SIZE = 4;
    // input

    bool oks = true;
    for(int j=0; j < 10; j++){

        int rand_value = utils::random::generate(0,12345);
        int *arr = utils::random::generate_array(1 + rand_value ,100 + rand_value, N);
        if(j<8) BLOCK_SIZE *= 2;
        DPRINT_ARR(arr, N)

        // reference implementation
        int *segmerge_sm_arr = new int[N];
        transposer::reference::segmerge_sm_step(arr, segmerge_sm_arr, N, BLOCK_SIZE);
        DPRINT_ARR(segmerge_sm_arr, N)

        // cuda implementation
        int *segmerge_sm_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_sm_cuda_out = utils::cuda::allocate<int>(N);
        transposer::cuda::segmerge_sm_step(segmerge_sm_cuda_in, segmerge_sm_cuda_out, N, BLOCK_SIZE);
        int *segmerge_sm_arr_2 = new int[N]; 
        utils::cuda::recv(segmerge_sm_arr_2, segmerge_sm_cuda_out, N);
        DPRINT_ARR(segmerge_sm_arr_2, N)

        bool ok = utils::equals<int>(segmerge_sm_arr, segmerge_sm_arr_2, N);
        oks = oks && ok;
        utils::cuda::deallocate(segmerge_sm_cuda_in);
        utils::cuda::deallocate(segmerge_sm_cuda_out);
        delete arr, segmerge_sm_arr, segmerge_sm_arr_2;


    }
    return oks;
}

// ================================================
//  segmerge3 sm 
bool transposer::component_test::segmerge3_sm_step() {

    const int N = 100;
    // input

    bool oks = true;
    int BLOCK_SIZE = 4;

    for(int j=0; j < 10; j++){
        int rand_value = utils::random::generate(0,12345);
        int *arr = utils::random::generate_array(1 + rand_value ,100 + rand_value, N);
        if(j<8) BLOCK_SIZE *= 2;
        DPRINT_ARR(arr, N)

        // reference implementation
        int *segmerge_sm_arr = new int[N];
        int *segmerge_a_in_arr = new int[N];
        int *segmerge_a_out_arr = new int[N];
        int *segmerge_b_in_arr = new int[N];
        int *segmerge_b_out_arr = new int[N];
        transposer::reference::segmerge3_sm_step(arr, segmerge_sm_arr, N, BLOCK_SIZE,
                                                segmerge_a_in_arr, segmerge_a_out_arr, segmerge_b_in_arr, segmerge_b_out_arr);

        DPRINT_ARR(segmerge_sm_arr, N)

        // cuda implementation
        int *segmerge_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_a_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_b_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_cuda_out = utils::cuda::allocate<int>(N);
        int *segmerge_a_cuda_out = utils::cuda::allocate<int>(N);
        int *segmerge_b_cuda_out = utils::cuda::allocate<int>(N);
    
        transposer::cuda::segmerge3_sm_step(segmerge_cuda_in, segmerge_cuda_out, N, BLOCK_SIZE,
                                            segmerge_a_cuda_in, segmerge_b_cuda_in, segmerge_a_cuda_out, segmerge_b_cuda_out);
        int *segmerge_sm_arr_2 = new int[N]; 
        utils::cuda::recv(segmerge_sm_arr_2, segmerge_cuda_out, N);
        DPRINT_ARR(segmerge_sm_arr_2, N)

        bool ok = utils::equals<int>(segmerge_sm_arr, segmerge_sm_arr_2, N);
        oks = oks && ok;
        utils::cuda::deallocate(segmerge_cuda_in);
        utils::cuda::deallocate(segmerge_a_cuda_in);
        utils::cuda::deallocate(segmerge_b_cuda_in);
        utils::cuda::deallocate(segmerge_cuda_out);
        utils::cuda::deallocate(segmerge_a_cuda_out);
        utils::cuda::deallocate(segmerge_b_cuda_out);
        delete arr, segmerge_sm_arr, segmerge_sm_arr_2, segmerge_a_in_arr, segmerge_a_out_arr, segmerge_b_in_arr, segmerge_b_out_arr;

    }
    return oks;
}

/*
// ===============================================================================
bool transposer::component_test::segmerge_static_sm() {

    const int N = 49;

    int *arr = utils::random::generate_array(1, 5, N);
    DPRINT_ARR(arr, N)

    // reference implementation
    int *segmerge_arr = new int[N];
    transposer::reference::segmerge_step(arr, segmerge_arr, N);
    DPRINT_ARR(segmerge_arr, N)

    // cuda implementation
    int *segmerge_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
    int *segmerge_cuda_out = utils::cuda::allocate<int>(N);
    transposer::cuda::segmerge3_step(segmerge_cuda_in, segmerge_cuda_out, N);
    int *segmerge_arr_2 = new int[N]; 
    utils::cuda::recv(segmerge_arr_2, segmerge_cuda_out, N);
    DPRINT_ARR(segmerge_arr_2, N)

    bool ok = utils::equals<int>(segmerge_arr, segmerge_arr_2, N);

    utils::cuda::deallocate(segmerge_cuda_in);
    utils::cuda::deallocate(segmerge_cuda_out);
    delete arr, segmerge_arr, segmerge_arr_2;

    return ok;
}


// ===============================================================================
bool transposer::component_test::segmerge3_static_sm() {

    const int N = 49;

    int *arr = utils::random::generate_array(1, 5, N);
    DPRINT_ARR(arr, N)

    // reference implementation
    int *segmerge_arr = new int[N];
    transposer::reference::segmerge_step(arr, segmerge_arr, N);
    DPRINT_ARR(segmerge_arr, N)

    // cuda implementation
    int *segmerge_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
    int *segmerge_cuda_out = utils::cuda::allocate<int>(N);
    transposer::cuda::segmerge3_step(segmerge_cuda_in, segmerge_cuda_out, N);
    int *segmerge_arr_2 = new int[N]; 
    utils::cuda::recv(segmerge_arr_2, segmerge_cuda_out, N);
    DPRINT_ARR(segmerge_arr_2, N)

    bool ok = utils::equals<int>(segmerge_arr, segmerge_arr_2, N);

    utils::cuda::deallocate(segmerge_cuda_in);
    utils::cuda::deallocate(segmerge_cuda_out);
    delete arr, segmerge_arr, segmerge_arr_2;

    return ok;
}
*/