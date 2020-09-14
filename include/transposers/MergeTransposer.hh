#pragma once

#include "AbstractTransposer.hh"
#include <cstring>
#include <algorithm>

#ifndef DIV_THEN_CEIL
#define DIV_THEN_CEIL(a,b) a%b?((a/b)+1):(a/b)
#endif

class sort_indices
{
    private:
        int* mparr;

    public:
        sort_indices(int* parr) : mparr(parr) {}
        bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
};

class MergeTransposer : public AbstractTransposer
{

private:

    

    const int N_THREAD = 100;

    virtual int csr2csc(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal) 
    {

        // 1. espando `rowPtr` per farlo diventare `rowIdx`
        int* rowIdx = new int[nnz];
        for(int j = 0; j < m; j++) {
            for(int i = csrRowPtr[j]; i < csrRowPtr[j+1]; i++) {
                rowIdx[i] = j;
            }
        }

        // 2. creo i vari blocchi
        int* blockColPtr = new int[N_THREAD * (n+1)](); // () inizializza a zero tutto

        // 3. riempio i vari blocchi - seriale
        int* colIdxSorted = new int[nnz];
        std::memcpy(colIdxSorted, csrColIdx, nnz*sizeof(int));

        int len = DIV_THEN_CEIL(nnz, N_THREAD);
        int* indices = new int[len];

        for(int i = 0; i < N_THREAD; i++) {

            int start_index = i*len;
            int end_index = (i+1)*len < nnz ? (i+1)*len : nnz;
            int current_len = end_index - start_index;

            // riordino i blocchi facendo il sort
            for(int j = 0; j < current_len; j++) { 
                indices[j] = j;
            }

            std::sort(indices+start_index, indices+end_index, sort_indices(indices));
            
            for(int j = start_index; j < end_index; j++) {
                colIdxSorted[j] = csrColIdx[indices[j]]; // OK perchè i due array sono diversi
                cscRowIdx[j] = rowIdx[indices[j]]; 
                cscVal[j] = csrVal[indices[j]];
            }

            // creo il `colPtr` dal `colIdxSorted` corrente
            int* currentBlockColPtr = blockColPtr + i*(nnz+1);

            for(int j = start_index; j < end_index; j++) {
                currentBlockColPtr[colIdxSorted[j] + 1]++;
            }
            for(int j = 1; j < n + 1; j++) { // prefix_sum
                currentBlockColPtr[j] += currentBlockColPtr[j-1];
            }

        }


        // deallocazione
        delete rowIdx;
        delete colIdxSorted;
        delete blockColPtr;
        delete indices;
    }

public:
    MergeTransposer() : AbstractTransposer() { }
};
