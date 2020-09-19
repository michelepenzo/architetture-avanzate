#pragma once

#include "AbstractTransposer.hh"
#include <cstring>
#include <algorithm>

#ifndef DIV_THEN_CEIL
#define DIV_THEN_CEIL(a,b) a%b?((a/b)+1):(a/b)
#endif

void compress_pointers(int n, int start_index, int end_index, int const idx[], int ptr[]) {

    for(int j = start_index; j < end_index; j++) { 
        ptr[idx[j] + 1]++;
    } 
    for(int j = 1; j < n + 1; j++) { // prefix_sum
        ptr[j] += idx[j-1];
    }
}

void compress_pointers_all(int n, int nnz, int n_thread, 
    int const * colIdx, int *blockColPtr) {

    int len = DIV_THEN_CEIL(nnz, n_thread);

    for(int i = 0; i < n_thread; i++) {

        int start_index = i*len;
        int end_index = (i+1)*len < nnz ? (i+1)*len : nnz;
        int current_len = end_index - start_index;

        int* colPtr = blockColPtr + (i * (n+1));
        compress_pointers(n, start_index, end_index, colIdx, colPtr);
    }
}

void sort_for_column(
    int nnz, int N_THREAD, 
    int const * colIdxIn, int const * rowIdxIn, float const * valIn,
    int *colIdxOut, int *rowIdxOut, float *valOut
) {

    class sort_indices {
    private:
        const int const * mparr;
    public:
        sort_indices(const int const * parr) : mparr(parr) { }
        bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
    };

    int len = DIV_THEN_CEIL(nnz, N_THREAD);
    int* indices = new int[len];

    for(int i = 0; i < N_THREAD; i++) {

        int start_index = i*len;
        int end_index = (i+1)*len < nnz ? (i+1)*len : nnz;
        int current_len = end_index - start_index;

        // riordino i blocchi facendo il sort
        for(int j = 0; j < current_len; j++) { indices[j] = j; }

        std::sort(indices+start_index, indices+end_index, sort_indices(colIdxIn));
        
        for(int j = start_index; j < end_index; j++) {
            colIdxOut[j] = colIdxIn[indices[j]]; // OK perchè i due array sono diversi
            rowIdxOut[j] = rowIdxIn[indices[j]]; 
            valOut[j] = valIn[indices[j]];
        }
    }

}

void merge_pcsc_serial(
    int begin, int end,
    int const colPtrA[], int const rowIdxA[], float const valA[],
    int const colPtrB[], int const rowIdxB[], float const valB[],
    int colPtrC[], int rowIdxC[], int valC[])
{

    std::cout << "begin: " << begin << "\nend: " << end << std::endl;
    for(int i = begin; i < end+1; i++) {
        colPtrC[i] = colPtrA[i] + colPtrB[i];
    }

    for(int i = begin; i < end; i++) {
        int sa = colPtrA[i], la = colPtrA[i+1] - sa;
        int sb = colPtrB[i], lb = colPtrB[i+1] - sb;
        int sc = colPtrC[i], lc = colPtrC[i+1] - sc;

        for(int j = 0; j < la; j++) {
            rowIdxC[sc+j] = rowIdxA[sa + j];
            valC[sc+j] = valA[sa + j];

            std::cout << "rowIdxC: " << rowIdxC[sc+j] << "\tvalC:" << val[sc+j];
        }

        std::cout << std::endl;
        sc = sc + la;
        for(int j = 0; j < la; j++) {
            rowIdxC[sc+j] = rowIdxC[sb + j];
            valC[sc+j] = valC[sb + j];

            std::cout << "rowIdxC: " << rowIdxC[sc+j] << "\tvalC:" << val[sc+j];
        }
        std::cout << std::endl;
    }
}

class MergeTransposer : public AbstractTransposer
{

private:

    struct merge_buffer {
        const int n, nnz, n_thread;
        int* blockColPtr;
        int* rowIdx;
        float* val;
        merge_buffer(int n, int nnz, int n_thread) : n(n), nnz(nnz), n_thread(n_thread) { }
        void allocate() {
            blockColPtr = new int[n_thread * (n+1)];
            rowIdx = new int[nnz];
            val = new float[nnz];
        }
        void deallocate() {
            delete blockColPtr, rowIdx, val;
        }
        int* getColPtr(int thread) {
            return blockColPtr + (thread * (n+1));
        }

        void print_blockColPtr(){
            for (int i = 0; i < n_thread*(n+1); ++i) std::cout << blockColPtr[i] << " ";
        }

        void print_rowIdx(){
            for (int i = 0; i < nnz; ++i) std::cout << rowIdx[i] << " ";
        }

        void print_val(){
            for (int i = 0; i < nnz; ++i) std::cout << val[i] << " ";
        }

    };

    const int N_THREAD = 100;

    virtual int csr2csc(
        int m, int n, int nnz, 
        int const csrRowPtr[], int const csrColIdx[], float const csrVal[], 
        int* cscColPtr, int* cscRowIdx, float* cscVal) 
    {
        merge_buffer ping(n, nnz, N_THREAD), pong(n, nnz, N_THREAD);
        ping.allocate();
        pong.allocate();

        // 1. espando `rowPtr` per farlo diventare `rowIdx`
        
        std::cout << "espansione rowPtr" << std::endl;
        for(int j = 0; j < m; j++) {
            for(int i = csrRowPtr[j]; i < csrRowPtr[j+1]; i++) {
                ping.rowIdx[i] = j;
                std::cout << "ping: " << ping.rowIdx[i] << std::endl;
            }
        }

        
        std::cout << "riempio blocchi" << srd::endl;
        // 2. riempio i vari blocchi - seriale
        {
            int* colIdxSorted = new int[nnz];
            std::memcpy(colIdxSorted, csrColIdx, nnz*sizeof(int));

            // sorting dei valori
            sort_for_column(nnz, N_THREAD, 
                csrColIdx, ping.rowIdx, csrVal,
                colIdxSorted, pong.rowIdx, pong.val);

            // riempio `blockColPtr` di pong
            compress_pointers_all(n, nnz, N_THREAD, colIdxSorted, pong.blockColPtr);

            std::coud << "pong.blockColPtr: " << std::endl;
            pong.print_blockColPtr;

            std::coud << std::endl << "pong.rowIdx: " << std::endl;
            pong.print_rowIdx;

            std::coud << std::endl << "pong.val: " << std::endl;
            pong.print_val;


            delete colIdxSorted;
        }

        // 3. `ping` vuoto, `pong` ha valori validi
        merge_buffer buffers[2] = { ping, pong };
        int full = 1;
        
        std::cout << "merge_buffer creato" << std::endl;

        // 4. chiamo il merge
        for(int nblocks = N_THREAD; nblocks >= 2; nblocks /= 2)

            std::cout << "chiamo merge, nblocks: " << nblocks << std::endl; 
            const int len = DIV_THEN_CEIL(nnz, nblocks);
            for(int i = 0; i < nblocks; i++) {
                std::cout << "\t i: " << i << std::endl;
                
                int begin = i * len;
                int end = (i+1) * len; if(end > nnz) { end = nnz; }

                merge_pcsc_serial(begin, end, 
                    buffers[full].blockColPtr + i*(n+1), buffers[full].rowIdx, buffers[full].val,
                    buffers[full].blockColPtr + (i+1)*(n+1), buffers[full].rowIdx, buffers[full].val,
                    buffers[1-full].blockColPtr + i*(n+1), buffers[1-full].rowIdx, buffers[1-full].val
                );
            }

            full = 1-full; // ping pong della memoria
        }

        std::cout << "deallocate"
        // deallocazione
        ping.deallocate();
        pong.deallocate();
    }

public:
    MergeTransposer() : AbstractTransposer() { }
};
