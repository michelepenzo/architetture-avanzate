#include "transposers.hh"

int transposers::serial_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {
    
    int* curr = new int[n](); // array inizializzato con tutti '0'

    DPRINT_MSG("Reference")
    DPRINT_ARR(csrRowPtr, m+1)
    DPRINT_ARR(csrColIdx, nnz)
    DPRINT_ARR(csrVal, nnz)

    // 1. costruisco `cscColPtr` come istogramma delle frequenze degli elementi per ogni colonna
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            cscColPtr[csrColIdx[j]+1]++;
        }
    }
    // 2. applico prefix_sum per costruire corretto `cscColPtr` (ogni cella tiene conto dei precedenti)
    for(int i = 1; i < n+1; i++) {
        cscColPtr[i] += cscColPtr[i-1];
    }
    // 3. sistemo indici di riga e valori
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            int col = csrColIdx[j];
            int loc = cscColPtr[col] + curr[col];
            curr[col]++;
            cscRowIdx[loc] = i;
            cscVal[loc] = csrVal[j];
        }
    }

    DPRINT_MSG("SERIAL")
    DPRINT_ARR(cscColPtr, n+1)
    DPRINT_ARR(cscRowIdx, nnz)
    DPRINT_ARR(   cscVal, nnz)

    delete[] curr;
    return COMPUTATION_OK;
}

int transposers::cuda_wrapper(
    int m, int n, int nnz,
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal,
    algo _algo
) {

    int * csrRowPtr_cuda = utils::cuda::allocate_send<int>(csrRowPtr, m+1);
    int * csrColIdx_cuda = utils::cuda::allocate_send<int>(csrColIdx, nnz);
    float * csrVal_cuda  = utils::cuda::allocate_send<float>(csrVal, nnz);

    int * cscColPtr_cuda = utils::cuda::allocate_zero<int>(n+1);
    int * cscRowIdx_cuda = utils::cuda::allocate_zero<int>(nnz);
    float * cscVal_cuda  = utils::cuda::allocate_zero<float>(nnz);

    int esito = _algo(
        m, n, nnz,
        csrRowPtr_cuda, csrColIdx_cuda, csrVal_cuda, 
        cscColPtr_cuda, cscRowIdx_cuda, cscVal_cuda
    );

    utils::cuda::deallocate(csrRowPtr_cuda);
    utils::cuda::deallocate(csrColIdx_cuda);
    utils::cuda::deallocate(csrVal_cuda);

    utils::cuda::deallocate_recv<int>(cscColPtr, cscColPtr_cuda, n+1);
    utils::cuda::deallocate_recv<int>(cscRowIdx, cscRowIdx_cuda, nnz);
    utils::cuda::deallocate_recv<float>(cscVal, cscVal_cuda, nnz);

    return esito;
}

__global__ 
void reorder_elements_kernel(
    int n, int nnz,
    int *inter, int *intra, 
    int *csrRowIdx, int *csrColIdx, float *csrVal,
    int *cscColPtr, int *cscRowIdx, float *cscVal
) {

    const int j = blockIdx.x;
    
    // allineo inter alla riga corretta
    inter += j*n;

    // recupero gli estremi della porzione di array da processare
    const int BLOCK_SIZE = DIV_THEN_CEIL(nnz, HISTOGRAM_BLOCKS);
    const int start = j*BLOCK_SIZE;
    const int end = min((j+1)*BLOCK_SIZE, nnz);
    const int len = end - start;

    // calcolo la posizione degli elementi
    for(int i = 0; i < len; i++) {
        int cid = csrColIdx[start + i];
        int loc = cscColPtr[cid] + inter[cid] + intra[i];
        cscRowIdx[loc] = csrRowIdx[start + i];
        cscVal[loc] = csrVal[start + i];
    }
}

int transposers::scan_csr2csc(
    int m, int n, int nnz,
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    // 1. espandi l'array di puntatori agli indici di riga, nell'array di indici esteso
    int * csrRowIdx = utils::cuda::allocate_zero<int>(nnz);
    procedures::cuda::pointers_to_indexes(csrRowPtr, m, csrRowIdx, nnz);

    // 2. riempi inter, intra, e colPtr
    int * inter;
    int * intra = utils::cuda::allocate_zero<int>(nnz);
    int * colPtr = utils::cuda::allocate_zero<int>(n+1);
    procedures::cuda::indexes_to_pointers(csrColIdx, nnz, &inter, intra, colPtr, n);

    // 3. applica scan ai puntatori
    procedures::cuda::scan(colPtr, cscColPtr, n+1);

    // 4. permuta valori
    reorder_elements_kernel<<<HISTOGRAM_BLOCKS, 1>>>(
        n, nnz,
        inter, intra, 
        csrRowIdx, csrColIdx, csrVal, 
        cscColPtr, cscRowIdx, cscVal);
    CUDA_CHECK_ERROR

    DPRINT_MSG("SCAN")
    DPRINT_ARR_CUDA(cscColPtr, n+1)
    DPRINT_ARR_CUDA(cscRowIdx, nnz)
    DPRINT_ARR_CUDA(cscVal, nnz)
    for(int i = 0; i <= HISTOGRAM_BLOCKS; i++) {
        DPRINT_ARR_CUDA(inter+i*n, n)
    }
    DPRINT_ARR_CUDA(intra, nnz)


    utils::cuda::deallocate(csrRowIdx);
    utils::cuda::deallocate(inter);
    utils::cuda::deallocate(intra);
    utils::cuda::deallocate(colPtr);
    return COMPUTATION_OK;
}

int transposers::merge_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    // alloca lo spazio necessario per effettuare il sort
    struct merge_buffer {
        //int * colPtrBlock;
        int * colIdx;
        int * rowIdx;
        float * val;
    };
    merge_buffer buffer[2];
    buffer[0].colIdx = utils::cuda::allocate_zero<int>(nnz);
    buffer[0].rowIdx = utils::cuda::allocate_zero<int>(nnz);
    buffer[0].val    = utils::cuda::allocate_zero<float>(nnz);
    buffer[1].colIdx = utils::cuda::allocate_zero<int>(nnz);
    buffer[1].rowIdx = cscRowIdx;
    buffer[1].val    = cscVal;
    int * colPtr     = utils::cuda::allocate_zero<int>(n+1);

    // 1. espandi rowPtr in rowIdx
    DPRINT_MSG("1 ---- row IDX to PTR")
    procedures::cuda::pointers_to_indexes(csrRowPtr, m, buffer[0].rowIdx, nnz);
    DPRINT_ARR_CUDA(buffer[0].rowIdx, nnz);

    // 2. ordina per indice delle colonne
    DPRINT_MSG("2 ---- sort by column index")
    utils::cuda::copy(buffer[0].colIdx, csrColIdx, nnz);
    procedures::cuda::segsort3(buffer[0].colIdx, buffer[1].colIdx, nnz, buffer[0].rowIdx, buffer[1].rowIdx, csrVal, buffer[1].val);
    DPRINT_ARR_CUDA(buffer[1].colIdx,  nnz);
    DPRINT_ARR_CUDA(buffer[1].rowIdx, nnz);
    DPRINT_ARR_CUDA(buffer[1].val,    nnz);

    // 3. merging
    DPRINT_MSG("3 ---- merging")
    int full = 1;
    int CURRENT_BLOCK_SIZE = SEGSORT_ELEMENTS_PER_BLOCK;

    while(CURRENT_BLOCK_SIZE < (nnz-1)*2) {
    
        procedures::cuda::segmerge3_step(
            buffer[full].colIdx, buffer[full].colIdx, 
            nnz, CURRENT_BLOCK_SIZE, 
            buffer[full].rowIdx, buffer[1-full].rowIdx, 
            buffer[full].val, buffer[1-full].val
        );
    
        full = 1 - full;
        CURRENT_BLOCK_SIZE *= 2;
        DPRINT_MSG("Block size %d", CURRENT_BLOCK_SIZE)
        DPRINT_ARR_CUDA(buffer[full].colIdx,  nnz);
        DPRINT_ARR_CUDA(buffer[full].rowIdx, nnz);
        DPRINT_ARR_CUDA(buffer[full].val,    nnz);
    }

    // 4. index to pointers
    DPRINT_MSG("4 ---- colIdx -> colPtr")
    int * inter;
    procedures::cuda::indexes_to_pointers(buffer[full].colIdx, nnz, &inter, colPtr, n);
    procedures::cuda::scan(colPtr, cscColPtr, n+1);

    if(full != 1) {
        utils::cuda::copy<int>(cscRowIdx, buffer[full].rowIdx, nnz);
        utils::cuda::copy<float>(cscVal, buffer[full].val, nnz);
    }

    DPRINT_ARR_CUDA(cscColPtr, n+1);
    DPRINT_ARR_CUDA(cscRowIdx, nnz);
    DPRINT_ARR_CUDA(cscVal,    nnz);

    utils::cuda::deallocate(inter);
    utils::cuda::deallocate(colPtr);
    utils::cuda::deallocate(buffer[0].colIdx);
    utils::cuda::deallocate(buffer[1].colIdx);
    utils::cuda::deallocate(buffer[0].rowIdx);
    utils::cuda::deallocate(buffer[0].val);

    return 0;
}