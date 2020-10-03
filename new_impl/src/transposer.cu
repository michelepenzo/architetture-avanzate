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
// SCAN ==========================================================================
// ===============================================================================

#define SCAN_THREAD_PER_BLOCK 512
#define SCAN_ELEMENTS_PER_BLOCK (2*SCAN_THREAD_PER_BLOCK)


__global__ 
void add_kernel(int INPUT_ARRAY array, int INPUT_ARRAY incs, int len) {

	int i = blockIdx.x * SCAN_ELEMENTS_PER_BLOCK + threadIdx.x;
	int b = blockIdx.x;
	if(i < len) {
        array[i] += incs[b];
    }
}

__global__
void scan_kernel(int INPUT_ARRAY input, int * output, int len, int * sums) {

    extern __shared__ int temp[]; // TODO allocazione FISSA ha incremento performace?

    int blockID = blockIdx.x;
	int blockOffset = blockID * SCAN_ELEMENTS_PER_BLOCK;
    int i = threadIdx.x;

    // caricamento dei dati in shared memory: ogni thread carica esattamente due elementi
    temp[2*i]   = (blockOffset + 2*i   < len) ? input[blockOffset + 2*i]   : 0;
    temp[2*i+1] = (blockOffset + 2*i+1 < len) ? input[blockOffset + 2*i+1] : 0;

    // Blelloch Scan
    int offset = 1;

    // prima parte dell'algoritmo: ogni elemento viene sommato
    // al successivo, poi a quello x2 in avanti, poi x4, ..., xD con D=log_2(powtwo)
    for(int d = SCAN_ELEMENTS_PER_BLOCK/2; d > 0; d = d/2) {
        __syncthreads();
        if(i < d) {
            int ai = offset * (2*i + 1) - 1;
			int bi = offset * (2*i + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
    }
    __syncthreads();

    // se sono il thread 0 allora metto a zero l'ultimo elemento
    if(i == 0) {
        sums[blockID] = temp[SCAN_ELEMENTS_PER_BLOCK - 1];
        temp[SCAN_ELEMENTS_PER_BLOCK-1] = 0;
    }

    // seconda parte dell'algoritmo: "downsweep"
    for (int d = 1; d < SCAN_ELEMENTS_PER_BLOCK; d *= 2) // traverse down tree & build scan
	{
		offset /= 2;
		__syncthreads();

		if (i < d)
		{
			int ai = offset * (2*i + 1) - 1;
			int bi = offset * (2*i + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

    __syncthreads();
    // scrivo in output i risultati
    if(2*i < len) { 
        output[blockOffset + 2*i] = temp[2*i];
    }
    if(2*i+1 < len) { 
        output[blockOffset + 2*i+1] = temp[2*i+1];
    }
}

void scan_small(int INPUT_ARRAY input, int * output, int len) {

    // TODO testare diff performance settando SUMS == null
    int *sums = utils::cuda::allocate_zero<int>(1);

    scan_kernel<<< 1, SCAN_THREAD_PER_BLOCK, 2 * SCAN_ELEMENTS_PER_BLOCK * sizeof(int) >>>(
        input, output, len, sums);
    CUDA_CHECK_ERROR

    utils::cuda::deallocate<int>(sums);
}

void scan_large(int INPUT_ARRAY input, int * output, int len) {
    
    const int BLOCKS = DIV_THEN_CEIL(len, SCAN_ELEMENTS_PER_BLOCK);
    int *sums = utils::cuda::allocate_zero<int>(BLOCKS);
    int *incs = utils::cuda::allocate_zero<int>(BLOCKS);
    
    // 1. chiamo il kernel
    scan_kernel<<< BLOCKS, SCAN_THREAD_PER_BLOCK, 2 * SCAN_ELEMENTS_PER_BLOCK * sizeof(int) >>>(
        input, output, len, sums);
    CUDA_CHECK_ERROR

    // 2. ricorsivamente applico scan a sums per ottenere l'array di incrementi
    transposer::cuda::scan(sums, incs, BLOCKS);

    // 3. ad ogni cella del blocco 'i' aggiungo l'incremento 'incs[i]'
    add_kernel<<< BLOCKS, SCAN_ELEMENTS_PER_BLOCK >>>(output, incs, len);
    CUDA_CHECK_ERROR

    utils::cuda::deallocate<int>(sums);
    utils::cuda::deallocate<int>(incs);
}


void transposer::cuda::scan(int INPUT_ARRAY input, int * output, int len) {

    if(len <= SCAN_ELEMENTS_PER_BLOCK) {
        // scan senza array di somme temporaneo
        scan_small(input, output, len);
    } else {
        // scan con array somme temporanee
        scan_large(input, output, len);
    }
}

void transposer::reference::scan(int INPUT_ARRAY input, int * output, int len) {

    output[0] = 0;
    for(int i = 1; i < len; i++) {
        output[i] = output[i-1] + input[i-1];
    }
}

// ===============================================================================
// SEG SORT ======================================================================
// ===============================================================================

#define SEGSORT_ELEMENTS_PER_BLOCK 4
// #define SEGSORT_ELEMENTS_PER_BLOCK 32

__global__
void segsort_kernel(int INPUT_ARRAY input, int * output, int len) {

    __shared__ int temp[SEGSORT_ELEMENTS_PER_BLOCK];
    int thread_id = threadIdx.x;
    int global_id = blockIdx.x * SEGSORT_ELEMENTS_PER_BLOCK + threadIdx.x;

    // caricamento dati in shared memory
    int element = (global_id < len) ? input[global_id] : INT32_MAX;
    temp[thread_id] = element;
    __syncthreads();

    /// trovo la posizione del `thread_id`-esimo elemento
    int index = 0;
    for(int i = 0; i < thread_id; i++) {
        if(temp[i] <= element) {
            index++;
        }
    }
    for(int i = thread_id+1; i < SEGSORT_ELEMENTS_PER_BLOCK; i++) {
        if(temp[i] < element) {
            index++;
        }
    }
    __syncthreads();

    // porto alla posizione corretta
    temp[index] = element;
    __syncthreads();

    // scaricamento dati in shared memory
    if(thread_id < len) {
        output[global_id] = temp[thread_id];
    }
}

void transposer::cuda::segsort(int INPUT_ARRAY input, int * output, int len) {

    segsort_kernel<<< DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK), SEGSORT_ELEMENTS_PER_BLOCK >>>(
        input, output, len
    );
    CUDA_CHECK_ERROR
}

void transposer::reference::segsort(int INPUT_ARRAY input, int * output, int len) {

    utils::copy_array(output, input, len);

    const int N = DIV_THEN_CEIL(len, SEGSORT_ELEMENTS_PER_BLOCK);
    for(int i = 0; i < N; i++) {
        const int start = i * SEGSORT_ELEMENTS_PER_BLOCK;
        const int end = std::min((i + 1) * SEGSORT_ELEMENTS_PER_BLOCK, len);
        std::sort(output + start, output + end); // TODO controlla comparatore default
    }
}

// ===============================================================================
// SEG SORT ======================================================================
// ===============================================================================

__global__
void merge_kernel(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {

    int global_id = blockIdx.x; // * SEGSORT_ELEMENTS_PER_BLOCK + threadIdx.x;
    int start_1 = 2 * global_id * BLOCK_SIZE;
    int start_2 = (2 * global_id + 1) * BLOCK_SIZE;
    int end_1 = min((2 * global_id + 1) * BLOCK_SIZE, len);
    int end_2 = min((2 * global_id + 2) * BLOCK_SIZE, len);

    //printf("%d: start1= %d, start2=%d, end1=%d, end2=%d\n", global_id, start_1, start_2, end_1, end_2);

    int current_1 = start_1;
    int current_2 = start_2;
    int current_output = start_1;
    
    // merge
    while(current_1 < end_1 && current_2 < end_2) {
        if(input[current_1] <= input[current_2]) {
            //printf("MERGE1 output[%d] = input[%d] = %d\n", current_output, current_1, input[current_1]);
            output[current_output] = input[current_1];
            current_1++;
        } else {
            //printf("MERGE2 output[%d] = input[%d] = %d\n", current_output, current_1, input[current_1]);
            output[current_output] = input[current_2];
            current_2++;
        }
        current_output++;
    }

    // finisco le rimanenze del primo blocco
    while(current_1 < end_1) {
        //printf("COPY1 output[%d] = input[%d] = %d\n", current_output, current_1, input[current_1]);
        output[current_output] = input[current_1];
        current_1++;
        current_output++;
    }

    // finisco le rimanenze del secondo blocco
    while(current_2 < end_2) {
        //printf("COPY2 output[%d] = input[%d] = %d\n", current_output, current_2, input[current_2]);
        output[current_output] = input[current_2];
        current_2++;
        current_output++;
    }
}

#define MERGE_SPLITTER_DISTANCE 4
// #define MERGE_SPLITTER_DISTANCE 256

__global__
void seg_splitter_kernel(int INPUT_ARRAY input, int * splitter, int len, const int BLOCK_SIZE) {

    int block_id = blockIdx.x;
    int th_id = threadIdx.x;

    // indici di inizio e fine della porzione di array di input
    int start_input = block_id * BLOCK_SIZE;
    // int end_input = min((block_id + 1) * BLOCK_SIZE, len);

    // indici di inizio e fine della porzione di array di splitter
    const int SPLITTER_BLOCK_SIZE = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_SPLITTER_DISTANCE);
    int start_splitter = block_id * SPLITTER_BLOCK_SIZE;

    // riempio l'array di splitter
    for(int i = th_id; i < SPLITTER_BLOCK_SIZE; i = i + blockDim.x) {
        //int offset = i * MERGE_SPLITTER_DISTANCE;
        //if(start_input + offset <= end_input) {
            splitter[start_splitter + i] = input[start_input + i * MERGE_SPLITTER_DISTANCE];
        //} else {
        //    break;
        //}        
    }
}


__device__
int find(int element_to_search, int INPUT_ARRAY input, int len) {

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

__global__
void seg_searchindex_kernel(int INPUT_ARRAY input, int INPUT_ARRAY splitter, int * indexA, int * indexB, int len, const int BLOCK_SIZE) {

    int block_id = blockIdx.x;
    int th_id = threadIdx.x;

    // indici di inizio e fine della porzione di array di input assegnata al blocco A, ed al blocco B
    int start_A = 2 * block_id * BLOCK_SIZE;
    int start_B = (2 * block_id + 1) * BLOCK_SIZE;
    int end_A = min((2 * block_id + 1) * BLOCK_SIZE, len);
    int end_B = min((2 * block_id + 2) * BLOCK_SIZE, len);

    // indici di inizio della porzione di array di splitter
    const int SPLITTER_PER_BLOCK = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_SPLITTER_DISTANCE);
    int start_splitter = 2 * block_id       * SPLITTER_PER_BLOCK;
    int end_splitter   = 2 * (block_id + 1) * SPLITTER_PER_BLOCK;

    // ogni thread si dedica ad un elemento di splitter
    for(int i = start_splitter + th_id; i < end_splitter; i = i + blockDim.x) {
        int element = splitter[i];
        indexA[i] = find(element, input + start_A, end_A - start_A);
        indexB[i] = find(element, input + start_B, end_B - start_B);
    }
}

__global__
void seg_splittermerge_kernel(int INPUT_ARRAY input, int INPUT_ARRAY indexA, int INPUT_ARRAY indexB, int * output, int len, const int BLOCK_SIZE) {

    // seg_splittermerge_kernel<<<BLOCK_NUMBER * (SPLITTER_PER_BLOCK + 1), MERGE_SPLITTER_DISTANCE>>>

    int i = blockIdx.x;

    // Ogni `i` (blockIdx.x) effettua un piccolo merge in memoria shared. 
    // Devo risalire a quale SUB-SPLITTER corrisponde l'elemento `i` per sapere se è
    // il primo o l'ultimo del SUB-SPLITTER 
    
    const int SPLITTER_PER_BLOCK = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_SPLITTER_DISTANCE);
    const int SPLITTER_PER_COUPLE_BLOCKS = 2 * SPLITTER_PER_BLOCK + 1;

    int which_item = i % SPLITTER_PER_COUPLE_BLOCKS;
    bool is_first_element = (which_item == 0);
    bool is_last_element = (which_item == SPLITTER_PER_COUPLE_BLOCKS-1);

    // prendo gli indici di inizio e fine del blocco in input contrassegnato 
    // da A, e B. Se sono il primo blocco parto da 0, se sono l'ultimo finisco con BLOCK_SIZE
    int startA = is_first_element ? 0 : indexA[i-1];
    int startB = is_first_element ? 0 : indexB[i-1];
    int endA   = is_last_element ? BLOCK_SIZE : indexA[i];
    int endB   = is_last_element ? BLOCK_SIZE : indexB[i];

    // Ogni griglia di thread ha una memoria condivisa sulla quale copiare gli elementi
    // dell'array di input fino ad un massimo di MERGE_SPLITTER_DISTANCE per blocco
    __shared__ int temp[2 * MERGE_SPLITTER_DISTANCE];

    // Carico le porzioni degli elementi
    int j = threadIdx.x;
    // elemento in input preso dall'elemento A della coppia di blocchi
    temp[j] = (startA + j < endA) ? input[startA + j] : 0;
    // elemento in input preso dall'elemento B della coppia di blocchi
    temp[MERGE_SPLITTER_DISTANCE + j] = (startB + j < endB) ? input[startB + j] : 0;
    // aspetto che temp sia stato completamente caricato
    __syncthreads();

    // cerco l'indice del mio elemento preso da A nella coppia di blocchi A, B mergiati
    int index1 = j + find(temp[j], temp + MERGE_SPLITTER_DISTANCE, endB - startB);       
    // cerco l'indice del mio elemento preso da B nella coppia di blocchi A, B mergiati
    int index2 = find(temp[MERGE_SPLITTER_DISTANCE + j], temp, endA - startA) + (j + MERGE_SPLITTER_DISTANCE); // elemento di B
    // aspetto che tutti abbiano cercato in temp
    __syncthreads();

    // salvo in output (start_A è l'inizio dell'intera coppia di blocchi)
    output[startA + index1] = temp[j];
    output[startA + index2] = temp[MERGE_SPLITTER_DISTANCE + j];

    printf("Block=%d, Thread=%d, A=[%d:%d], B=[%d:%d], elementA=%d->%d, elementB=%d->%d\n", 
        blockIdx.x, j, startA, endA, startB, endB, temp[j], index1, temp[MERGE_SPLITTER_DISTANCE + j], index2
    );
}

void merge(int INPUT_ARRAY input, int * output, int len, const int BLOCK_SIZE);

void merge_big(int INPUT_ARRAY input, int * output, int len, const int BLOCK_SIZE) {

    const int SPLITTER_PER_BLOCK = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_SPLITTER_DISTANCE);
    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
    const int SPLITTER_NUMBER = BLOCK_NUMBER * SPLITTER_PER_BLOCK;

    int * splitter        = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * splitter_sorted = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * indexA          = utils::cuda::allocate<int>(SPLITTER_NUMBER);
    int * indexB          = utils::cuda::allocate<int>(SPLITTER_NUMBER);

    // 1. faccio lo split degli elementi 
    // Uso una grid per ogni sub-splitter array, ogni griglia può avere il numero di thread che mi pare per dividere il lavoro di copia
    seg_splitter_kernel<<<BLOCK_SIZE, 32>>>(input, splitter, len, BLOCK_SIZE);
    CUDA_CHECK_ERROR
    DPRINT_ARR_CUDA(splitter, DIV_THEN_CEIL(len, MERGE_SPLITTER_DISTANCE))

    // 2. faccio il merge degli elementi di splitter
    // Faccio il merge di `splitter` che è lungo SPLITTER_NUMBER elementi
    // divisi in BLOCK_NUMBER blocchi da SPLITTER_PER_BLOCK elementi.
    DPRINT_MSG("Merging splitter with len=%d, block_size=%d", SPLITTER_NUMBER, SPLITTER_PER_BLOCK)
    merge(splitter, splitter_sorted, SPLITTER_NUMBER, SPLITTER_PER_BLOCK);
    DPRINT_ARR_CUDA(splitter_sorted, SPLITTER_PER_BLOCK)

    // 3. cerco gli indici di ogni elemento
    // Utilizzo una GRIGLIA[di thread] ogni 2 blocchi.
    // Ogni griglia ha un numero arbitrario (es. 32) di thread che ci lavorano
    seg_searchindex_kernel<<<DIV_THEN_CEIL(BLOCK_SIZE, 2), 32>>>(input, splitter_sorted, indexA, indexB, len, BLOCK_SIZE);
    CUDA_CHECK_ERROR
    DPRINT_ARR_CUDA(indexA, DIV_THEN_CEIL(len, MERGE_SPLITTER_DISTANCE))
    DPRINT_ARR_CUDA(indexB, DIV_THEN_CEIL(len, MERGE_SPLITTER_DISTANCE))

    // 4. ogni thread fa ora il merge delle porzioni di A, B associate a splitter[i]
    // Ogni GRIGLIA[di thread] processa il merge di un numero massimo di elementi. 
    // Devo avere BLOCK_NUMBER * (SPLITTER_PER_BLOCK + 1) griglie perchè ad ogni blocco
    // sono associato SPLITTER_PER_BLOCK segmenti + uno finale
    // La griglia di thread viene dimensionata MERGE_SPLITTER_DISTANCE così che ogni thread
    // processa esattamente un elemento
    const int SPLITTER_PER_COUPLE_BLOCKS = 2 * SPLITTER_PER_BLOCK + 1;
    const int COUPLE_OF_BLOCKS = DIV_THEN_CEIL(BLOCK_NUMBER, 2);

    seg_splittermerge_kernel<<<COUPLE_OF_BLOCKS * (SPLITTER_PER_COUPLE_BLOCKS + 1), MERGE_SPLITTER_DISTANCE>>>(
        input, indexA, indexB, output, len, BLOCK_SIZE
    );
    CUDA_CHECK_ERROR
    DPRINT_ARR_CUDA(output, len)

    utils::cuda::deallocate(splitter);
    utils::cuda::deallocate(indexA);
    utils::cuda::deallocate(indexB);
}

void merge(int INPUT_ARRAY input, int * output, int len, const int INIT_BLOCK_SIZE) {

    int* buffer[2];
    buffer[0] = output;
    buffer[1] = utils::cuda::allocate<int>(len);
    int full = 0;

    // applico merge
    for(int BLOCK_SIZE = INIT_BLOCK_SIZE; BLOCK_SIZE < len; BLOCK_SIZE *= 2) {
        
        DPRINT_MSG("Block size=%d", BLOCK_SIZE)
        if(BLOCK_SIZE > 1) {
merge_big(buffer[full], buffer[1-full], len, BLOCK_SIZE);
        }
        

        full = 1 - full;
        DPRINT_ARR_CUDA(buffer[full], len)
    }

    // eventualmente copio nell'array di output nel caso non sia stato l'ultimo
    // ad essere riempito...
    if(full != 0) {
        utils::cuda::copy(output, buffer[1], len);
    }

    // dealloco array temporaneo;
    utils::cuda::deallocate(buffer[1]);



    //if(2 * BLOCK_SIZE > 2 * MERGE_SPLITTER_DISTANCE) {
    //    // due blocchi non rientrano nella shared memory allora uso il metodo `big merge`
    //    DPRINT_MSG("Block size=%d", BLOCK_SIZE)
    //    merge_big(input, output, len, BLOCK_SIZE);
    //
    //} else {
    //    // entrambi i blocchi insieme stanno nella shared memory
    //    DPRINT_MSG("Small merge: block size=%d", BLOCK_SIZE)
    //    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
    //    merge_kernel<<< DIV_THEN_CEIL(BLOCK_NUMBER, 2), 1 >>>(input, output, len, BLOCK_SIZE);
    //    CUDA_CHECK_ERROR
    //    DPRINT_ARR_CUDA(output, len)
    //}
}

void transposer::cuda::sort(int INPUT_ARRAY input, int * output, int len) {

    transposer::cuda::segsort(input, output, len);

    merge(input, output, len, SEGSORT_ELEMENTS_PER_BLOCK);
}

void transposer::reference::sort(int INPUT_ARRAY input, int * output, int len) {

    utils::copy_array(output, input, len);
    std::sort(output, output + len);
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

bool transposer::component_test::scan() {

    const int N = 1000000;
    // input
    int *arr = utils::random::generate_array(1, 1, N);
    DPRINT_ARR(arr, N)

    // reference implementation
    int *scan_arr = new int[N];
    transposer::reference::scan(arr, scan_arr, N);
    DPRINT_ARR(scan_arr, N)

    // cuda implementation
    int *scan_arr_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
    int *scan_arr_cuda_out = utils::cuda::allocate_zero<int>(N);
    transposer::cuda::scan(scan_arr_cuda_in, scan_arr_cuda_out, N);
    int *scan_arr_2 = new int[N]; 
    utils::cuda::recv(scan_arr_2, scan_arr_cuda_out, N);
    DPRINT_ARR(scan_arr_2, N)

    bool ok = utils::equals<int>(scan_arr, scan_arr_2, N);

    utils::cuda::deallocate(scan_arr_cuda_in);
    utils::cuda::deallocate(scan_arr_cuda_out);
    delete arr, scan_arr, scan_arr_2;
    
    return ok;
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