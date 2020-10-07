#include "procedures.hh"

__device__
void copy(int * output, int INPUT_ARRAY input, int len) {
    for(int i = 0; i < len; i++) {
        output[i] = input[i];
    }
}

__device__
int find_position_in_sorted_array(int element_to_search, int INPUT_ARRAY input, int len) {

    if(len <= 0) {
        return 0;
    }

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
int find_last_position_in_sorted_array(int element_to_search, int INPUT_ARRAY input, int len) {

    int index = find_position_in_sorted_array(element_to_search, input, len);
    while(input[index] == element_to_search && index < len) {
        //printf("input[%d] = %d == %d, index++\n", index, input[index], element_to_search);
        index++;
    }
    return index;
}

__global__
void splitter_kernel(int INPUT_ARRAY input, int * splitter, int * indexA, int * indexB, int len, int BLOCK_SIZE) {

    const int couple_block_id = blockIdx.x;

    // entrambi inputA, inputB esistono (eventualmente B ha lunghezza < BLOCK_SIZE se ultimo blocco)
    int * inputA = input + 2 * couple_block_id * BLOCK_SIZE;
    int * inputB = input + (2 * couple_block_id + 1) * BLOCK_SIZE;
    int lenA = BLOCK_SIZE;
    int lenB = min((2 * couple_block_id + 2) * BLOCK_SIZE, len) - (2 * couple_block_id + 1) * BLOCK_SIZE;

    // mi sposto verso gli indici corretti di splitter, indexA, indexB
    const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, SEGMERGE_SM_SPLITTER_DISTANCE);
    splitter = splitter + 2 * couple_block_id * SPLITTER_PER_BLOCKS;
    indexA = indexA + 2 * couple_block_id * SPLITTER_PER_BLOCKS;
    indexB = indexB + 2 * couple_block_id * SPLITTER_PER_BLOCKS;

    // riempio gli elementi
    int i;
    for(int i = 0; i < SPLITTER_PER_BLOCKS; i ++) {
        splitter[i] = inputA[i*SEGMERGE_SM_SPLITTER_DISTANCE];
        indexA[i] = i*SEGMERGE_SM_SPLITTER_DISTANCE;
        indexB[i] = find_position_in_sorted_array(splitter[i], inputB, lenB);
    }

    for(i = 0; i < SPLITTER_PER_BLOCKS && i*SEGMERGE_SM_SPLITTER_DISTANCE < lenB; i ++) {
        int element = inputB[i*SEGMERGE_SM_SPLITTER_DISTANCE];
        // save splitter
        splitter[SPLITTER_PER_BLOCKS + i] = element;
        indexA[SPLITTER_PER_BLOCKS + i] = find_last_position_in_sorted_array(element, inputA, lenA);
        indexB[SPLITTER_PER_BLOCKS + i] = i*SEGMERGE_SM_SPLITTER_DISTANCE;
        //printf("(%2d): element %d from B is position %d of A\n", couple_block_id, element, indexA[SPLITTER_PER_BLOCKS + i]);
    }
}

__global__
void fix_indexes_kernel(int * indexA, int * indexB, int len, int BLOCK_SIZE, int SPLITTER_NUMBER, int SPLITTER_PER_BLOCK) {

    int couple_block_id = blockIdx.x;

    if(threadIdx.x == 0) {

        // calcolo gli indici di inizio e fine degli splitter che devo processare
        int startSplitter = 2 * couple_block_id * SPLITTER_PER_BLOCK;
        int endSplitter  = min(2 * (couple_block_id + 1) * SPLITTER_PER_BLOCK, SPLITTER_NUMBER);
        int lenSplitter = endSplitter - startSplitter;
        // la lunghezza di A è sempre BLOCK_SIZE, la lunghezza di B?... dipende se è l'ultimo
        int lenB = min(2 * (couple_block_id + 1) * BLOCK_SIZE, len) - ((2 * couple_block_id + 1) * BLOCK_SIZE);
        
        printf("(%2d) split[%3d:%3d] {max %d}\n", blockIdx.x, startSplitter, endSplitter, SPLITTER_NUMBER);
    
        for(int i = 0; i < lenSplitter - 1; i += 1) {
            indexA[startSplitter+i] = indexA[startSplitter+i+1];
            indexB[startSplitter+i] = indexB[startSplitter+i+1];
        }

        // l'ultimo elemento contiene la dimensione del blocco
        indexA[lenSplitter-1] = BLOCK_SIZE;
        indexB[lenSplitter-1] = lenB;
        __syncthreads();
    }
}

__global__
void uniform_merge_kernel(int INPUT_ARRAY input, int * output, int INPUT_ARRAY indexA, int INPUT_ARRAY indexB, int len, int BLOCK_SIZE) {

    __shared__ int temp_in[2 * SEGMERGE_SM_SPLITTER_DISTANCE];
    __shared__ int temp_out[2 * SEGMERGE_SM_SPLITTER_DISTANCE];

    for(int i = 0; i < 2 * SEGMERGE_SM_SPLITTER_DISTANCE; i++) {
        temp_in[i] = 0;
        temp_out[i] = 0;
    }

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
    if(endA - startA > SEGMERGE_SM_SPLITTER_DISTANCE) printf("!!!Error A[%d:%d] > %d SM SP S\n", startA, endA, SEGMERGE_SM_SPLITTER_DISTANCE);
    copy(temp_in, inputA + startA, endA - startA);
    if(endB - startB > SEGMERGE_SM_SPLITTER_DISTANCE) printf("!!!Error B[%d:%d] > %d SM SP S\n", startB, endB, SEGMERGE_SM_SPLITTER_DISTANCE);
    copy(temp_in + SEGMERGE_SM_SPLITTER_DISTANCE, inputB + startB, endB - startB);
    __syncthreads();

    // effettuo merge
    for(int i = 0; i < endA - startA; i++) {
        int element = temp_in[i];
        int posInA = i;
        int posInB = find_position_in_sorted_array(element, temp_in + SEGMERGE_SM_SPLITTER_DISTANCE, endB - startB);
        int k = posInA + posInB;
        temp_out[k] = element;
    }
    __syncthreads();

    for(int i = 0; i < endB - startB; i++) {
        int element = temp_in[SEGMERGE_SM_SPLITTER_DISTANCE + i];
        int k = i + find_last_position_in_sorted_array(element, temp_in, endA - startA);
        temp_out[k] = element;
    }
    __syncthreads();

    // salva output
    output = output + 2 * couple_block_id * BLOCK_SIZE;
    for(int i = 0; i < (endA - startA) + (endB - startB); i++) {
        output[startA + startB + i] = temp_out[i];
    }
}

void procedures::cuda::segmerge_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {
    
    DPRINT_MSG("\n\n\n##### Starting segmerge_sm_step")
    //DPRINT_ARR_CUDA(input, len)

    // 1. lavoro su coppie di blocchi per estrarre gli splitter e gli indici necessari a lavorarci sopra
    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
    const int COUPLE_OF_BLOCKS = BLOCK_NUMBER / 2;
    const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, SEGMERGE_SM_SPLITTER_DISTANCE);
    const int SPLITTER_PER_LAST_BLOCK = (len % BLOCK_SIZE == 0) ? SPLITTER_PER_BLOCKS : DIV_THEN_CEIL(len % BLOCK_SIZE, SEGMERGE_SM_SPLITTER_DISTANCE);
    const int SPLITTER_NUMBER = 
        (BLOCK_NUMBER%2==1)                      ? // il numero di blocchi da processare è dispari?
        (2*COUPLE_OF_BLOCKS*SPLITTER_PER_BLOCKS) : // se si, tutti i blocchi sa processare hanno dimensione piena
        (2*(COUPLE_OF_BLOCKS-1)*SPLITTER_PER_BLOCKS+SPLITTER_PER_BLOCKS+SPLITTER_PER_LAST_BLOCK); // se no, l'ultimo blocco da processare potrebbe avere lunghezza minore

    printf("SPLITTER_NUMBER=%d, SPLITTER_PER_BLOCK=%d, SPLITTER_PER_LAST_BLOCK=%d, BLOCK_NUMBER=%d, COUPLE_OF_BLOCKS=%d\n", 
        SPLITTER_NUMBER, SPLITTER_PER_BLOCKS, SPLITTER_PER_LAST_BLOCK, BLOCK_NUMBER, COUPLE_OF_BLOCKS);

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

    // 3. sistemo gli indici di `indexA`, `indexB` per evitare merge vuoti
    fix_indexes_kernel<<<COUPLE_OF_BLOCKS, 1>>>(indexA_out, indexB_out, len, BLOCK_SIZE, SPLITTER_NUMBER, SPLITTER_PER_BLOCKS);
    CUDA_CHECK_ERROR
    DPRINT_MSG("After fix_indexes_kernel")
    DPRINT_ARR_CUDA(splitter_out, SPLITTER_NUMBER)
    DPRINT_ARR_CUDA(indexA_out, SPLITTER_NUMBER)
    DPRINT_ARR_CUDA(indexB_out, SPLITTER_NUMBER)

    // 4. eseguo il merge di porzioni di blocchi di dimensione uniforme
    uniform_merge_kernel<<<SPLITTER_NUMBER, 1>>>(input, output, indexA_out, indexB_out, len, BLOCK_SIZE);
    CUDA_CHECK_ERROR

    // 5. eventualmente copio il risultato dell' ultimo blocco di array rimasto spaiato
    if(BLOCK_NUMBER % 2 == 1) {
        utils::cuda::copy<int>(
            output + 2 * COUPLE_OF_BLOCKS * BLOCK_SIZE, 
            input +  2 * COUPLE_OF_BLOCKS * BLOCK_SIZE, 
            len - 2 * COUPLE_OF_BLOCKS * BLOCK_SIZE
        );
    }

    //DPRINT_MSG("After uniform_merge_kernel")
    //DPRINT_ARR_CUDA(input, len)
    //DPRINT_ARR_CUDA(output, len)

    utils::cuda::deallocate(splitter);
    utils::cuda::deallocate(indexA);
    utils::cuda::deallocate(indexB);
    utils::cuda::deallocate(splitter_out);
    utils::cuda::deallocate(indexA_out);
    utils::cuda::deallocate(indexB_out);
}

void procedures::cuda::segmerge3_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {
    segmerge3_step(input, output, len, BLOCK_SIZE, a_in, a_out, b_in, b_out);
}

void procedures::reference::segmerge_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {
    segmerge_step(input, output, len, BLOCK_SIZE);
}

void procedures::reference::segmerge3_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {
    segmerge3_step(input, output, len, BLOCK_SIZE, a_in, a_out, b_in, b_out);
}
