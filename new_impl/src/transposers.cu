#include "transposers.hh"

int transposers::serial_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    int* curr = new int[n](); // array inizializzato con tutti '0'

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

    delete[] curr;
    return COMPUTATION_OK;
}

int transposers::scan_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    // 1. espandi l'array di puntatori agli indici di riga, nell'array di indici esteso
    int * csrRowIdx = utils::cuda::allocate_zero<int>(nnz);
    procedures::cuda::pointers_to_indexes(csrRowPtr, m, csrRowIdx, nnz);

    

}