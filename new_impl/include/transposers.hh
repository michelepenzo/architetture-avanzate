#pragma once
#ifndef TRANSPOSERS_HH_
#define TRANSPOSERS_HH_

#include "procedures.hh"

namespace transposers {

    typedef void (*algo)(int, int, int, int*, int*, float*, int*, int*, float*);

    void serial_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);

    void cuda_wrapper(
        int m, int n, int nnz,
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal,
        algo _algo);

    void scan_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);

    void merge_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);
}

#endif