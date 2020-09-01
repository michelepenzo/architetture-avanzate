#pragma once

#include <iostream>
#include <iomanip>
#include <set>
#include <random>
#include <chrono>
#include "AbstractMatrix.hh"

class SparseMatrix : public AbstractMatrix {

private:

    void fill_with_rand_numbers() {

        // 1. init random library and seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> index_distrib(0, m*n-1);
        std::uniform_int_distribution<int> values_distrib(1, 100);

        // 2. generate indices
        std::set<int> indices; // set prevents duplicate insertion
        while(indices.size() < nnz) {
            indices.insert(index_distrib(generator));
        }

        // 3. fill values
        int i = 0;
        for(const int& index : indices) {

            int col = index % n;
            int row = index / n;
            int val = values_distrib(generator);

            std::cout << "Generate row " << row << " col " << col << " value " << val << std::endl;

            csrRowPtr[row+1]++;
            csrColIdx[i] = col;
            csrVal[i] = val;
            i++;
        }

        // 4. prefix_sum on csrRowPtr
        for(int i = 1; i <= m; i++) {
            csrRowPtr[i] += csrRowPtr[i-1];
        }
    }

public:

    int* csrRowPtr;

    int* csrColIdx;

    float* csrVal;

    SparseMatrix(const int m, const int n, const int nnz, MatrixInitialization mi = RANDOM_INITIALIZATION) : AbstractMatrix(m, n, nnz) { 

        this->csrRowPtr = new int[this->m+1]();
        this->csrColIdx = new int[this->nnz]();
        this->csrVal = new float[this->nnz]();

        if(mi == RANDOM_INITIALIZATION) {
            fill_with_rand_numbers();
        }
    }

    ~SparseMatrix() {
        delete[] csrRowPtr;
        delete[] csrColIdx;
        delete[] csrVal;
    }

    inline void print() {
        std::cout << std::endl << "csrRowPtr: ";
        for(int i = 0; i < m + 1; i++) {
            std::cout << csrRowPtr[i] << " ";
        }

        std::cout << std::endl << "csrColIdx: ";
        for(int i = 0; i < nnz; i++) {
            std::cout << csrColIdx[i] << " ";
        }

        std::cout << std::endl << "csrVal: ";
        for(int i = 0; i < nnz; i++) {
            std::cout << csrVal[i] << " ";
        }

        std::cout << std::endl;
    }

};