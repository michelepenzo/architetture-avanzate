#pragma once
#ifndef MATRIX_HH_
#define MATRIX_HH_

#include <iostream>
#include <iomanip>
#include <set>
#include "utilities.hh"

#define SPARSE_MATRIX_MIN_VAL 1
#define SPARSE_MATRIX_MAX_VAL 100

namespace matrix {

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
            if(m <= 0 || n <= 0 || nnz <= 0 || nnz > n * m) {
                throw std::invalid_argument("received negative value");
            }

            this->csrRowPtr = new int[this->m+1]();
            //DPRINT_MSG("Allocating csrRowPtr: %p", this->csrRowPtr)
            this->csrColIdx = new int[this->nnz]();
            //DPRINT_MSG("Allocating csrColIdx: %p", this->csrColIdx)
            this->csrVal = new float[this->nnz]();
            //DPRINT_MSG("Allocating csrVal: %p", this->csrVal)

            if(mi == RANDOM_INITIALIZATION) {

                // 1. generate indices
                std::set< std::tuple<int, int> > indices; // set prevents duplicate insertion
                
                while(indices.size() < nnz) {
                    std::tuple<int, int> t = std::make_tuple<int, int>(
                        utils::random::generate(m-1), 
                        utils::random::generate(n-1)
                    );
                    indices.insert(t);
                }

                // 2. fill values
                int i = 0;
                for(const std::tuple<int, int>& index : indices) {

                    int row = std::get<0>(index);
                    int col = std::get<1>(index);
                    int val = utils::random::generate(SPARSE_MATRIX_MIN_VAL, SPARSE_MATRIX_MAX_VAL);

                    ASSERT_LIMIT(row, m)
                    ASSERT_LIMIT(col, n)
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
            //DPRINT_MSG("Deallocating csrRowPtr: %p", csrRowPtr)
            delete[] csrRowPtr;
            //DPRINT_MSG("Deallocating csrColIdx: %p", csrColIdx)
            delete[] csrColIdx;
            //DPRINT_MSG("Deallocating csrVal: %p", csrVal)
            delete[] csrVal;
            //DPRINT_MSG("End deallocating")
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