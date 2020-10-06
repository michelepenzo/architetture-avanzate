#include "transposer.hh"

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
			temp[ai] = temp[bi];bool transposer::component_test::scan() {

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