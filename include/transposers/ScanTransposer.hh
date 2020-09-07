#pragma once

#include "CudaTransposer.hh"

#define SCANTRANS_DEBUG_ENABLE 1

class ScanTransposer : public CudaTransposer
{
protected:

    int csr2csc_gpumemory(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, float *csrVal, int *cscColPtr, int *cscRowIdx, float *cscVal);

public:

    ScanTransposer() : CudaTransposer() {}
};