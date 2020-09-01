#pragma once

#include "AbstractTransposer.hh"
#include "../CheckError.cuh"
#include <cuda_runtime.h>


class CudaTransposer : public AbstractTransposer {

protected:

    inline int csr2csc(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal) {

        int* csrRowPtr_dev;
        int* csrColIdx_dev;
        float* csrVal_dev;
        int* cscColPtr_dev;
        int* cscRowIdx_dev;
        float* cscVal_dev;

        cudaMalloc(&csrRowPtr_dev,(m+1)*sizeof(int));
        cudaMalloc(&csrColIdx_dev,(nnz)*sizeof(int));
        cudaMalloc(&csrVal_dev   ,(nnz)*sizeof(float));
        cudaMalloc(&cscColPtr_dev,(n+1)*sizeof(int));
        cudaMalloc(&cscRowIdx_dev,(nnz)*sizeof(int));
        cudaMalloc(&cscVal_dev   ,(nnz)*sizeof(float));

        SAFE_CALL(cudaMemcpy(csrRowPtr_dev, csrRowPtr, (m+1)*sizeof(int), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(csrColIdx_dev, csrColIdx, (nnz)*sizeof(int), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(csrVal_dev,    csrVal,    (nnz)*sizeof(int), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(cscColPtr_dev, cscColPtr, (n+1)*sizeof(int), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(cscRowIdx_dev, cscRowIdx, (nnz)*sizeof(int), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(cscVal_dev,    cscVal,    (nnz)*sizeof(int), cudaMemcpyHostToDevice));

        int ret = csr2csc_gpumemory(m, n, nnz, csrRowPtr_dev, csrColIdx_dev, csrVal_dev, cscColPtr_dev, cscRowIdx_dev, cscVal_dev);

        SAFE_CALL(cudaMemcpy(csrRowPtr, csrRowPtr_dev, (m+1)*sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(csrColIdx, csrColIdx_dev, (nnz)*sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(csrVal,    csrVal_dev,    (nnz)*sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(cscColPtr, cscColPtr_dev, (n+1)*sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(cscRowIdx, cscRowIdx_dev, (nnz)*sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(cscVal,    cscVal_dev,    (nnz)*sizeof(int), cudaMemcpyDeviceToHost));

        cudaFree(csrRowPtr_dev);
        cudaFree(csrColIdx_dev);
        cudaFree(csrVal_dev);
        cudaFree(cscColPtr_dev);
        cudaFree(cscRowIdx_dev);
        cudaFree(cscVal_dev);

        return ret;
    }

    virtual int csr2csc_gpumemory(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal) = 0;

public:

    CudaTransposer(SparseMatrix* sm) : AbstractTransposer(sm) { }

};