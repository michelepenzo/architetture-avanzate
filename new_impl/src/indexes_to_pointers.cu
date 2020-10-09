#include "procedures.hh"

__global__ 
void parallel_histogram_kernel(int INPUT_ARRAY idx, int idx_len, int * inter, int * intra, int HISTO_ROW_LEN) {
    
    int j = blockIdx.x;
    int t = threadIdx.x;
    
    // allineo inter sulla porzione di array che sto usando
    const int BLOCK_SIZE = DIV_THEN_CEIL(idx_len, HISTOGRAM_BLOCKS);
    const int START = BLOCK_SIZE * j;
    const int END = min(BLOCK_SIZE * (j+1), idx_len);

    if(t < END - START) {
        // ogni [blocco di thread] segna i contributi di una [porzione di idx]
        for(int i = START + t; i < END; i += blockDim.x) {
            int index = idx[i];
            intra[i] = inter[HISTO_ROW_LEN * (j+1) + index];
            atomicAdd(inter + HISTO_ROW_LEN * (j+1) + index, 1);
        }
    }
}

__global__
void vertical_scan_kernel(int * inter, int * ptr, int HISTO_ROW_LEN) {

    __shared__ int temp[HISTOGRAM_BLOCKS+1];

    int j = blockIdx.x;

    // caricamento dei dati
    for(int i = 0; i < HISTOGRAM_BLOCKS; i++) {
        temp[i] = inter[HISTO_ROW_LEN * i + j];
    }
    temp[HISTOGRAM_BLOCKS] = 0;

    // applico prefix scan
    for(int i = 0; i < HISTOGRAM_BLOCKS; i++) {
        temp[i+1] += temp[i];
    }

    // salvo i dati
    for(int i = 0; i < HISTOGRAM_BLOCKS; i++) {
        inter[HISTO_ROW_LEN * (i+1) + j] += temp[i];
    }
    ptr[j] = inter[HISTO_ROW_LEN*HISTOGRAM_BLOCKS + j];
}

void procedures::cuda::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int ** inter, int * intra, int * ptr, int ptr_len) {

    *inter = utils::cuda::allocate_zero<int>((HISTOGRAM_BLOCKS+1) * ptr_len);

    parallel_histogram_kernel<<<HISTOGRAM_BLOCKS, 1024>>>(idx, idx_len, *inter, intra, ptr_len);
    CUDA_CHECK_ERROR

    vertical_scan_kernel<<<ptr_len, 1>>>(*inter, ptr, ptr_len);
    CUDA_CHECK_ERROR
}

void procedures::reference::indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int ** _inter, int * intra, int * ptr, int ptr_len) {
    
    int * inter = new int[(HISTOGRAM_BLOCKS+1) * ptr_len]();
    *_inter = inter;

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
