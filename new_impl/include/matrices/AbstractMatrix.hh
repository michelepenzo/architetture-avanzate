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
        
        // check nnz correctness
        int the_nnz = nnz;
        if( nnz > m*n ) {
            std::cerr << "Number of non-zero element is " << nnz << " higher than " << m*n << " cells in "
                      << m << "*" << n << " matrix. NNZ is set to " << m*n << std::endl;
            the_nnz = m*n;
        }

        this->m = m;
        this->n = n;
        this->nnz = the_nnz;
    }

    virtual void print() = 0;

};
