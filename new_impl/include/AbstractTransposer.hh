#pragma once

#include "SparseMatrix.hh"
#include "Timer.cuh"
using namespace timer;

template<timer_type ttype> class AbstractTransposer {

private:

    virtual int csr2csc(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal) = 0;

    Timer<ttype> timer;

public:

    AbstractTransposer() : timer() { }


};

class SerialTransposer : public AbstractTransposer<HOST> {

};