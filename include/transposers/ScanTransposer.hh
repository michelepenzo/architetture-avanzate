#pragma once

#include "CudaTransposer.hh"

#define SCANTRANS_DEBUG_ENABLE 1

class ScanTransposer : public CudaTransposer
{
protected:

    const int BLOCK_SIZE;

    const int N_THREAD;

    int csr2csc_gpumemory(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, float *csrVal, int *cscColPtr, int *cscRowIdx, float *cscVal);

private: 

    void csrrowidx_caller(int m, int *csrRowPtr, int *cscRowIdx);

    void inter_intra_caller(int n, int nnz, int *inter, int *intra, int *csrColIdx);

    void vertical_scan_caller(int n, int *inter, int *cscColPtr);

    void prefix_sum(int n, int *cscColPtr);

    void reorder_elements_caller(
        int n, int nnz, int *inter, int *intra, 
        int *csrRowIdx, int *csrColIdx, float *csrVal,
        int *cscColPtr, int *cscRowIdx, float *cscVal
    );

public:

    ScanTransposer(int blocksize=256, int nthread=256) 
        : CudaTransposer(), BLOCK_SIZE(blocksize), N_THREAD(nthread) {}
};