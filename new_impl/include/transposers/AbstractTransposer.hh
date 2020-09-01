#pragma once

#include "../matrices/SparseMatrix.hh"
#define COMPUTATION_ERROR -1
#define COMPUTATION_OK 0

class AbstractTransposer {

protected:

    SparseMatrix* sm;

    virtual int csr2csc(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal) = 0;

public:

    AbstractTransposer(SparseMatrix* sm) : sm(sm) { }

    inline SparseMatrix* transpose() {

        SparseMatrix* result = new SparseMatrix(sm->n, sm->m, sm->nnz, ALL_ZEROS_INITIALIZATION);

        int esito = csr2csc(sm->m, sm->n, sm->nnz, 
                sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
                result->csrRowPtr, result->csrColIdx, result->csrVal);

        if(esito == COMPUTATION_ERROR) {
            delete result;
            return NULL;

        } else {
            return result;
        } 
    }

};