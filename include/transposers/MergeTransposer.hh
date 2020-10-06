#pragma once

#include "AbstractTransposer.hh"
#include <cstring>
#include <algorithm>

#ifndef DIV_THEN_CEIL
#define DIV_THEN_CEIL(a,b) a%b?((a/b)+1):(a/b)
#endif

void compress_pointers(int n, int len, int const idx[], int ptr[]) {

    std::cout << "compress_pointers: idx=";
    for(int i = 0; i < len; i++) std::cout << idx[i] << " ";
    std::cout << "\n";

    for(int j = 0; j < len; j++) { 
        ptr[idx[j] + 1]++;
    } 
    for(int j = 1; j < n + 1; j++) { // prefix_sum
        ptr[j] += ptr[j-1];
    }

    std::cout << "compress_pointers: ptr=";
    for(int i = 0; i < n+1; i++) std::cout << ptr[i] << " ";
    std::cout << "\n";
}

void compress_pointers_all(int n, int nnz, int n_thread, 
    int const * const colIdx, int *blockColPtr) {

    int len = DIV_THEN_CEIL(nnz, n_thread);

    for(int i = 0; i < n_thread; i++) {

        int start_index = i*len;
        int end_index = (i+1)*len < nnz ? (i+1)*len : nnz;
        int current_len = end_index - start_index;

        std::cout << "compress_pointers_all: idx from=" << start_index << " len=" << current_len << "\n";
        std::cout << "compress_pointers_all: thread=" << i << "\n";

        int* colPtr = blockColPtr + (i * (n+1));
        compress_pointers(n, current_len, colIdx+start_index, colPtr);
    }
}

void sort_for_column_partial(
    int begin, int end,
    int const * const colIdxIn, int const * const rowIdxIn, float const * const valIn,
    int *colIdxOut, int *rowIdxOut, float *valOut
) {

    class sort_indices {
    private:
        int const * const mparr;
    public:
        sort_indices(int const * const parr) : mparr(parr) { }
        bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
    };

    int len = end - begin;
    int* indices = new int[len];
    for(int i = 0; i < len; i++) indices[i] = i;

    std::cout << "sort_for_column_partial: Sort from " << begin << " to " << end << " (len=" << len << ")\n";

    // ordina indici
    std::sort(indices, indices+len, sort_indices(colIdxIn+begin));
    std::cout << "sort_for_column_partial: Indices: ";
    for(int i = 0; i < len; i++) std::cout << indices[i] << " ";
    std::cout << "\n";

    for(int j = 0; j < len; j++) {
        colIdxOut[begin+j] = colIdxIn[begin+indices[j]]; // OK perchè i due array sono diversi
        rowIdxOut[begin+j] = rowIdxIn[begin+indices[j]]; 
        valOut[begin+j] = valIn[begin+indices[j]];
    }
}

void sort_for_column(
    int nnz, int N_THREAD, 
    int const * colIdxIn, int const * rowIdxIn, float const * valIn,
    int *colIdxOut, int *rowIdxOut, float *valOut
) {

    

    int len = DIV_THEN_CEIL(nnz, N_THREAD);
    int* indices = new int[len]();

    for(int i = 0; i < N_THREAD; i++) {

        int start_index = i*len;
        int end_index = (i+1)*len < nnz ? (i+1)*len : nnz;
        int current_len = end_index - start_index;

        if(current_len > 0) {
            std::cout << "sort_for_column: Sorting thread " << i << ": start=" << start_index 
                << ", end=" << end_index << ", len=" << current_len << "\n";

            sort_for_column_partial(start_index, end_index,
                colIdxIn, rowIdxIn, valIn,
                colIdxOut, rowIdxOut, valOut);
        } else {
            //std::cout << "Sorting thread " << i << "skip\n";
        }        
    }

}

void merge_pcsc_serial(
    int begin, int end,
    int colPtrA[], int rowIdxA[], float valA[],
    int colPtrB[], int rowIdxB[], float valB[],
    int colPtrC[], int rowIdxC[], float valC[])
{

    std::cout << "\tmerge_pcsc_serial: begin: " << begin << " end: " << end << std::endl;
    std::cout << "\tmerge_pcsc_serial: colPtrC: ";
    for(int i = begin; i < end+1; i++) {
        colPtrC[i] = colPtrA[i] + colPtrB[i];
        std::cout << colPtrC[i] << " ";
    }
    std::cout << "\n";

    for(int i = begin; i < end; i++) {
        int sa = colPtrA[i], la = colPtrA[i+1] - sa;
        int sb = colPtrB[i], lb = colPtrB[i+1] - sb;
        int sc = colPtrC[i]; // lc = colPtrC[i+1] - sc;

        std::cout << "\tmerge_pcsc_serial: i=" << i << std::endl;
        std::cout << "\tmerge_pcsc_serial: merging from A (la=" << la << ")" << std::endl;
        for(int j = 0; j < la; j++) {
            rowIdxC[sc+j] = rowIdxA[sa + j];
            valC[sc+j] = valA[sa + j];

            std::cout << "\t\ttmerge_pcsc_serial: j=" << j <<" rowIdxC=" << rowIdxC[sc+j] << " valC=" << valC[sc+j] << std::endl;
        }

        sc = sc + la;
        std::cout << "\tmerge_pcsc_serial: merging from B (lb=" << lb << ")" << std::endl;
        for(int j = 0; j < lb; j++) {
            rowIdxC[sc+j] = rowIdxC[sb + j];
            valC[sc+j] = valC[sb + j];

            std::cout << "\t\tmerge_pcsc_serial: j=" << j <<" rowIdxC=" << rowIdxC[sc+j] << " valC=" << valC[sc+j] << std::endl;
        }
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
            blockColPtr = new int[n_thread * (n+1)]();
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
            for (int i = 0; i < n_thread; ++i) {
                std::cout << "Thread " << i << ": ";
                for(int j = 0; j < n+1; j++) {
                    std::cout << blockColPtr[i*(n+1) + j] << " ";
                }
                std::cout << "\n";
            }
        }

        void print_rowIdx(){
            for (int i = 0; i < nnz; ++i) std::cout << rowIdx[i] << " ";
        }

        void print_val(){
            for (int i = 0; i < nnz; ++i) std::cout << val[i] << " ";
        }

    };

    const int N_THREAD = 2;

    int csr2csc(
        int m, int n, int nnz, 
        int *csrRowPtr, int *csrColIdx, float *csrVal, 
        int *cscColPtr, int *cscRowIdx, float *cscVal) 
    {
        merge_buffer ping(n, nnz, N_THREAD), pong(n, nnz, N_THREAD);
        ping.allocate();
        pong.allocate();

        // 1. espando `rowPtr` per farlo diventare `rowIdx`
        
        std::cout << "csr2csc: espansione rowPtr: " << std::endl;
        for(int j = 0; j < m; j++) {
            for(int i = csrRowPtr[j]; i < csrRowPtr[j+1]; i++) {
                ping.rowIdx[i] = j;
                std::cout << ping.rowIdx[i] << " ";
            }
        }

        
        // 2. riempio i vari blocchi - seriale
        {

            std::cout << "csr2csc: creo copia di colIdx" << std::endl;
            int* colIdxSorted = new int[nnz];
            std::memcpy(colIdxSorted, csrColIdx, nnz*sizeof(int));

            // sorting dei valori
            std::cout << "csr2csc: inizio sorting" << std::endl;
            sort_for_column(nnz, N_THREAD, 
                csrColIdx, ping.rowIdx, csrVal,
                colIdxSorted, pong.rowIdx, pong.val);

            std::cout << "csr2csc: stampa valori sorted:" << std::endl;
            std::cout << "csr2csc: csrColIdx: ";
            for(int i = 0; i < nnz; i++) std::cout << csrColIdx[i] << " ";
            std::cout << "\ncsr2csc: colIdxSorted: ";
            for(int i = 0; i < nnz; i++) std::cout << colIdxSorted[i] << " ";
            std::cout << "\ncsr2csc: row idx: ";
            pong.print_rowIdx();
            std::cout << "\ncsr2csc: val: ";
            pong.print_val();
            std::cout << "\n";

            // riempio `blockColPtr` di pong
            std::cout << "csr2csc: Riempio matrice blockColPtr\n";
            compress_pointers_all(n, nnz, N_THREAD, colIdxSorted, pong.blockColPtr);

            std::cout << std::endl << "csr2csc: pong.blockColPtr: " << std::endl;
            pong.print_blockColPtr();

            std::cout << std::endl << "csr2csc: pong.rowIdx: " << std::endl;
            pong.print_rowIdx();

            std::cout << std::endl << "csr2csc: pong.val: " << std::endl;
            pong.print_val();

            delete colIdxSorted;
        }

        // 3. `ping` vuoto, `pong` ha valori validi
        merge_buffer buffers[2] = { ping, pong };
        int full = 1;
        
        std::cout << "\ncsr2csc: starting merge..." << std::endl;

        // 4. chiamo il merge
        for(int nblocks = N_THREAD; nblocks > 1; nblocks /= 2) {

            std::cout << "csr2csc: chiamo merge, nblocks: " << nblocks << std::endl; 
            const int len = DIV_THEN_CEIL(nnz, nblocks);
            for(int i = 0; i < nblocks/2; i++) {
                std::cout << "csr2csc: i=" << i << std::endl;
                
                int begin = i * len;
                int end = (i+1) * len; if(end > nnz) { end = nnz; }

                merge_pcsc_serial(begin, end, 
                    buffers[full].blockColPtr + 2*i*(n+1), buffers[full].rowIdx, buffers[full].val,
                    buffers[full].blockColPtr + (2*i+1)*(n+1), buffers[full].rowIdx, buffers[full].val,
                    buffers[1-full].blockColPtr + 2*i*(n+1), buffers[1-full].rowIdx, buffers[1-full].val
                );
            }

            full = 1-full; // ping pong della memoria
        }

        std::cout << "csr2csc: deallocate\n";
        
        // deallocazione
        ping.deallocate();
        pong.deallocate();
        return 0;
    }

public:
    MergeTransposer() : AbstractTransposer() { }
};
