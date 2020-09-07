#pragma once

#include <iostream>
#include <iomanip>

enum MatrixInitialization {
    ALL_ZEROS_INITIALIZATION = 0,
    RANDOM_INITIALIZATION = 1
};

class AbstractMatrix {

protected:

    virtual void fill_with_rand_numbers() = 0;

public:
    
    int m, n, nnz;

    AbstractMatrix(const int m, const int n, const int nnz) {
        
        this->m = m;
        this->n = n;
        this->nnz = nnz;
    }

    virtual void print() = 0;

};
