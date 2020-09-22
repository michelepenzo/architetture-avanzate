#pragma once
#ifndef MATRIX_HH_
#define MATRIX_HH_

#include <iostream>
#include <iomanip>
#include <set>
#include <random>
#include <chrono>
#include "utilities.hh"

namespace matrix {

    const unsigned SEED = 1234567; // std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(SEED);
    const int MIN_VALUE = SPARSE_MATRIX_MIN_VAL, MAX_VALUE = SPARSE_MATRIX_MAX_VAL;

    enum MatrixInitialization {
        ALL_ZEROS_INITIALIZATION = 0,
        RANDOM_INITIALIZATION = 1
    };

    class SparseMatrix {
    public:

        const int m, n, nnz;
        int *csrRowPtr, *csrColIdx;
        float *csrVal;

        SparseMatrix(const int m, const int n, const int nnz, const MatrixInitialization mi = RANDOM_INITIALIZATION) 
            : m(m), n(n), nnz(nnz)
        { 

            this->csrRowPtr = new int[this->m+1]();
            this->csrColIdx = new int[this->nnz]();
            this->csrVal = new float[this->nnz]();

            if(mi == RANDOM_INITIALIZATION) {

                std::uniform_int_distribution<long long> index_distrib(0, ((long long)m) * ((long long)n) - 1);
                std::uniform_int_distribution<int> values_distrib(MIN_VALUE, MAX_VALUE);

                // 1. generate indices
                std::set<long long> indices; // set prevents duplicate insertion
                while(indices.size() < nnz) {
                    indices.insert(index_distrib(generator));
                }

                // 2. fill values
                int i = 0;
                for(const long long& index : indices) {

                    int col = (int)(index % n);
                    int row = (int)(index / n);
                    int val = values_distrib(generator);

                    ASSERT_LIMIT(col, n)
                    ASSERT_LIMIT(row, m)
                    ASSERT_RANGE(val)

                    csrRowPtr[row+1]++;
                    csrColIdx[i] = col;
                    csrVal[i] = val;
                    i++;
                }

                // 3. prefix_sum on csrRowPtr
                utils::prefix_sum(csrRowPtr, m+1);
            }
        }

        ~SparseMatrix() {
            delete[] csrRowPtr;
            delete[] csrColIdx;
            delete[] csrVal;
        }

        bool equals(SparseMatrix* sm) {
            return m == sm->m && n == sm->n && nnz == sm->nnz 
                && utils::equals(csrRowPtr, sm->csrRowPtr, m+1)
                && utils::equals(csrColIdx, sm->csrColIdx, nnz)
                && utils::equals(csrVal, sm->csrVal, nnz);
        }

        void print() {
            utils::print("csrRowPtr", csrRowPtr, m+1);
            utils::print("csrColIdx", csrColIdx, nnz);
            utils::print("   csrVal", csrVal, nnz);
            printf("%p %p %p\n", csrRowPtr, csrColIdx, csrVal);
        }

    };

}

#endif