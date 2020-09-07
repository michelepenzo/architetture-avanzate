#pragma once

#include "CudaTransposer.hh"

class ScanTransposer : public CudaTransposer {

protected:

    int csr2csc_gpumemory(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal) {

        return 0;
    }

public:

    ScanTransposer(SparseMatrix* sm) : CudaTransposer(sm) { }

};