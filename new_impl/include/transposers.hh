#pragma once
#ifndef TRANSPOSERS_HH_
#define TRANSPOSERS_HH_

#include "procedures.hh"

namespace transposers {

    int serial_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);

    int scan_csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal);

}





#endif