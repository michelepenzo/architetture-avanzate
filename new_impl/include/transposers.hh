#pragma once
#ifndef TRANSPOSERS_HH_
#define TRANSPOSERS_HH_

#include "procedures.hh"

namespace transposers {

    typedef int (*algo)(int, int, int, int*, int*, float*, int*, int*, float*);

    int serial_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);

    int cuda_wrapper(
        int m, int n, int nnz,
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal,
        algo _algo);

    int scan_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);

    int merge_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);
}

#endif