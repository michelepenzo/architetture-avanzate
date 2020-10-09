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

int scan_csr2csc_cuda(
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
    procedures::cuda::scan(colPtr, cscColPtr, n);

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

    utils::cuda::deallocate(csrRowIdx);
    utils::cuda::deallocate(inter);
    utils::cuda::deallocate(intra);
    utils::cuda::deallocate(colPtr);
    return COMPUTATION_OK;
}

int transposers::scan_csr2csc(

    int m, int n, int nnz,
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    int * csrRowPtr_cuda = utils::cuda::allocate_send<int>(csrRowPtr, m+1);
    int * csrColIdx_cuda = utils::cuda::allocate_send<int>(csrColIdx, nnz);
    float * csrVal_cuda  = utils::cuda::allocate_send<float>(csrVal, nnz);

    int * cscColPtr_cuda = utils::cuda::allocate_zero<int>(n+1);
    int * cscRowIdx_cuda = utils::cuda::allocate_zero<int>(nnz);
    float * cscVal_cuda  = utils::cuda::allocate_zero<float>(nnz);

    int esito = scan_csr2csc_cuda(
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