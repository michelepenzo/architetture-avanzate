#pragma once
#ifndef TRANSPOSER_HH_
#define TRANSPOSER_HH_

#include <algorithm>
#include "utilities.hh"
#include "matrix.hh"

#define HISTOGRAM_BLOCKS 2

#define SCAN_THREAD_PER_BLOCK 512
#define SCAN_ELEMENTS_PER_BLOCK (2*SCAN_THREAD_PER_BLOCK)

#define SEGSORT_ELEMENTS_PER_BLOCK 4

#define SEGMERGE_SM_SPLITTER_DISTANCE 4

namespace transposer {

    namespace cuda {

        void indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len);

        void pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len);
    
        void scan(int INPUT_ARRAY input, int * output, int len);

        void segsort(int INPUT_ARRAY input, int * output, int len);

        void segmerge_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE);

        void segmerge3_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out);

        void segmerge_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE);

        void segmerge3_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out);

        void sort(int INPUT_ARRAY input, int * output, int len);
    }

    namespace reference {
        
        void indexes_to_pointers(int INPUT_ARRAY idx, int idx_len, int * ptr, int ptr_len);

        void pointers_to_indexes(int INPUT_ARRAY ptr, int ptr_len, int * idx, int idx_len);

        void scan(int INPUT_ARRAY input, int * output, int len);

        void segsort(int INPUT_ARRAY input, int * output, int len);

        void segmerge_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE);

        void segmerge3_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out);

        void segmerge_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE);

        void segmerge3_sm_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out);

        void sort(int INPUT_ARRAY input, int * output, int len);
    }

    namespace component_test {

        bool indexes_to_pointers();

        bool pointers_to_indexes();

        bool scan();

        bool segsort();

        bool sort();

        bool segmerge_step();

        bool segmerge3_step();

        bool segmerge_sm_step();

        bool segmerge3_sm_step();
    }
    
    /*
    void sort_by_column(int len, 
        int INPUT_ARRAY colIdxIn, int INPUT_ARRAY rowIdxIn, float INPUT_ARRAY valIn,
        int *colIdxOut, int *rowIdxOut, float *valOut
    ) {
        if(len <= 0) {
            DPRINT_MSG("Length < 0")
            return;
        }

        // this structure helps ordering using only a single auxiliary array
        struct sort_indices {
            int const * const mparr;
            sort_indices(int const * const parr) : mparr(parr) { }
            bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
        };
        DPRINT_ARR(colIdxIn, len);

        // sorting
        int *indexes = utils::create_indexes(len);
        std::sort(indexes, indexes+len, sort_indices(colIdxIn));
        DPRINT_ARR(indexes, len);

        // permutation of the elements
        for(int j = 0; j < len; j++) {
            ASSERT_LIMIT(indexes[j], len)
            colIdxOut[j] = colIdxIn[indexes[j]];
            rowIdxOut[j] = rowIdxIn[indexes[j]]; 
            valOut[j] = valIn[indexes[j]];
        }
        DPRINT_ARR(colIdxOut, len);
        DPRINT_ARR(rowIdxOut, len);
        DPRINT_ARR(valOut, len);

        delete indexes;
    }


    struct merge_buffer {
        const int n, nnz, n_thread;
        int *blockColPtr, *rowIdx;
        float* val;
        merge_buffer(int n, int nnz, int n_thread) : n(n), nnz(nnz), n_thread(n_thread) { 
            blockColPtr = new int[n_thread * (n+1)]();
            rowIdx = new int[nnz]();
            val = new float[nnz]();
        }
        ~merge_buffer() {
            if(blockColPtr != NULL) { delete blockColPtr; blockColPtr = NULL; }
            if(rowIdx != NULL) { delete rowIdx; rowIdx = NULL; }
            if(val != NULL) { delete val; val = NULL; }
        }
        void reset() {
            DPRINT_MSG("Reset buffer...")
            std::memset(blockColPtr, 0, n_thread * (n+1) * sizeof(int));
            std::memset(     rowIdx, 0,              nnz * sizeof(int));
            std::memset(        val, 0,              nnz * sizeof(float));
        }
    };

    void merge_pcsc_2(
        int n, int nnz,
        int INPUT_ARRAY colPtrA, int INPUT_ARRAY rowIdxA, float INPUT_ARRAY valA,
        int INPUT_ARRAY colPtrB, int INPUT_ARRAY rowIdxB, float INPUT_ARRAY valB,
        int * colPtrC, int * rowIdxC, float * valC)
    {

        for(int i = 0; i < n+1; i++) {
            ASSERT_LIMIT(colPtrA[i], nnz) // overflow limit check
            ASSERT_LIMIT(colPtrB[i], nnz) // overflow limit check
            colPtrC[i] = colPtrA[i] + colPtrB[i];
        }

        for(int i = 0; i < n; i++) {

            int sa = colPtrA[i], la = colPtrA[i+1] - sa;
            int sb = colPtrB[i], lb = colPtrB[i+1] - sb;
            int sc = colPtrC[i], lc = colPtrC[i+1] - sc;

            DPRINT_MSG("Col=%d, sa=%d, sb=%d, sc=%d, la=%d, lb=%d, lc=%d", i, sa, sb, sc, la, lb, lc)

            utils::copy_array(rowIdxC+sc, rowIdxA+sa, la);
            utils::copy_array(   valC+sc,    valA+sa, la);

            sc = sc + la;
            utils::copy_array(rowIdxC+sc, rowIdxB+sb, lb);
            utils::copy_array(   valC+sc,    valB+sb, lb);
        }

        const int firstA = colPtrA[0], lastA = colPtrA[n];
        const int firstB = colPtrB[0], lastB = colPtrB[n];
        const int firstC = colPtrC[0], lastC = colPtrC[n];
        DPRINT_MSG("A = %d ... %d", firstA, lastA)
        DPRINT_MSG("B = %d ... %d", firstB, lastB)
        DPRINT_ARR(colPtrA, n + 1)
        DPRINT_ARR(colPtrB, n + 1)
        DPRINT_ARR(colPtrC, n + 1)
        DPRINT_ARR(rowIdxA, lastA - firstA)
        DPRINT_ARR(rowIdxB, lastB - firstB)
        DPRINT_ARR(rowIdxC, lastC - firstC)
        DPRINT_ARR(valA, lastA - firstA)
        DPRINT_ARR(valB, lastB - firstB)
        DPRINT_ARR(valC, lastC - firstC)
    }

    void merge_pcsc_3(
        int n, int nnz,
        int INPUT_ARRAY colPtrX, int INPUT_ARRAY rowIdxX, float INPUT_ARRAY valX,
        int INPUT_ARRAY colPtrY, int INPUT_ARRAY rowIdxY, float INPUT_ARRAY valY,
        int INPUT_ARRAY colPtrZ, int INPUT_ARRAY rowIdxZ, float INPUT_ARRAY valZ,
        int * colPtrC, int * rowIdxC, float * valC)
    {

        for(int i = 0; i < n+1; i++) {
            ASSERT_LIMIT(colPtrX[i], nnz) // overflow limit check
            ASSERT_LIMIT(colPtrY[i], nnz) // overflow limit check
            ASSERT_LIMIT(colPtrZ[i], nnz) // overflow limit check
            colPtrC[i] = colPtrX[i] + colPtrY[i] + colPtrZ[i];
        }

        for(int i = 0; i < n; i++) {

            int sx = colPtrX[i], lx = colPtrX[i+1] - sx;
            int sy = colPtrY[i], ly = colPtrY[i+1] - sy;
            int sz = colPtrZ[i], lz = colPtrZ[i+1] - sz;
            int sc = colPtrC[i], lc = colPtrC[i+1] - sc;

            DPRINT_MSG("Col=%d, sx=%d, sy=%d, sz=%d, sc=%d, lx=%d, ly=%d, lz=%d, lc=%d", i, sx, sy, sz, sc, lx, ly, lz, lc)

            utils::copy_array(rowIdxC+sc, rowIdxX+sx, lx);
            utils::copy_array(   valC+sc,    valX+sx, lx);

            sc = sc + lx;
            utils::copy_array(rowIdxC+sc, rowIdxY+sy, ly);
            utils::copy_array(   valC+sc,    valY+sy, ly);

            sc = sc + ly;
            utils::copy_array(rowIdxC+sc, rowIdxZ+sz, lz);
            utils::copy_array(   valC+sc,    valZ+sz, lz);
        }

        const int firstX = colPtrX[0], lastX = colPtrZ[n];
        const int firstY = colPtrY[0], lastY = colPtrY[n];
        const int firstZ = colPtrZ[0], lastZ = colPtrZ[n];
        const int firstC = colPtrC[0], lastC = colPtrC[n];
        DPRINT_MSG("X = %d ... %d", firstX, lastX)
        DPRINT_MSG("Y = %d ... %d", firstY, lastY)
        DPRINT_MSG("Z = %d ... %d", firstZ, lastZ)
        DPRINT_ARR(colPtrX, n + 1)
        DPRINT_ARR(colPtrY, n + 1)
        DPRINT_ARR(colPtrZ, n + 1)
        DPRINT_ARR(colPtrC, n + 1)
        DPRINT_ARR(rowIdxX, lastX - firstX)
        DPRINT_ARR(rowIdxY, lastY - firstY)
        DPRINT_ARR(rowIdxZ, lastZ - firstZ)
        DPRINT_ARR(rowIdxC, lastC - firstC)
        DPRINT_ARR(valX, lastX - firstX)
        DPRINT_ARR(valY, lastY - firstY)
        DPRINT_ARR(valZ, lastZ - firstZ)
        DPRINT_ARR(valC, lastC - firstC)
    }


    int merge_host_csr2csc(
        const int N_THREAD, const int m, const int n, const int nnz, 
        int INPUT_ARRAY csrRowPtr, int INPUT_ARRAY csrColIdx, float INPUT_ARRAY csrVal, 
        int *cscColPtr, int *cscRowIdx, float *cscVal
    ) {
        merge_buffer * buffer[2];
        buffer[0] = new merge_buffer(n, nnz, N_THREAD);
        buffer[1] = new merge_buffer(n, nnz, N_THREAD);

        // 1. expand `rowPtr` to `rowIdx`
        DPRINT_MSG("row pointer expands to indexes")
        DPRINT_ARR(csrRowPtr, m+1)
        pointers_to_indexes(csrRowPtr, m, buffer[0]->rowIdx, nnz);
        DPRINT_ARR(buffer[0]->rowIdx, nnz)

        // 2. sort by column indexes
        const int LEN = DIV_THEN_CEIL(nnz, N_THREAD);
        int* colIdxSorted = utils::copy_array(csrColIdx, nnz);

        for(int i = 0; i < N_THREAD; i++) {
            int start = i * LEN, end = std::min((i+1)*LEN, nnz);
            DPRINT_MSG("Sorting thread=%d sorting with start=%d, length=%d", i, start, end - start)

            sort_by_column(end - start, 
                csrColIdx+start,    buffer[0]->rowIdx+start, csrVal+start,
                colIdxSorted+start, buffer[1]->rowIdx+start, buffer[1]->val+start);
        }

        // 3. fill `buffer[1]` block col ptr 
        for(int i = 0; i < N_THREAD; i++) {
            int start = i * LEN, end = std::min((i+1)*LEN, nnz);
            DPRINT_MSG("Filling blocks of thread=%d with start=%d, length=%d", i, start, end - start)
            indexes_to_pointers(colIdxSorted+start, end-start, buffer[1]->blockColPtr + i*(n+1), n+1);
            utils::prefix_sum(buffer[1]->blockColPtr + i*(n+1), n+1);
        }

        // 4. call merge until only a single block remains
        int full = 1;
        int CLEN = LEN;
        for(int nblocks = N_THREAD; nblocks > 1; nblocks /= 2) {

            const bool IS_BLOCK_ODD = nblocks % 2 == 1;
            DPRINT_MSG("Joining %d blocks | Full buffer is=%d | len=%d", nblocks, full, CLEN)

            buffer[1-full]->reset();

            for(int i = 0 ; i < nblocks/2; i++) {

                if(IS_BLOCK_ODD && (i == nblocks/2-1)) {
                    DPRINT_MSG("Join %d + %d + %d", 2*i, 2*i+1, 2*i+2)
                    merge_pcsc_3( 
                        n, nnz,
                        buffer[full]->blockColPtr + 2*i*(n+1),       // X colptr
                        buffer[full]->rowIdx + 2*i*CLEN,             // X rowidx
                        buffer[full]->val + 2*i*CLEN,                // X val
                        buffer[full]->blockColPtr + (2*i+1)*(n+1),   // Y colptr
                        buffer[full]->rowIdx + (2*i+1)*CLEN,         // Y rowidx
                        buffer[full]->val + (2*i+1)*CLEN,            // Y val
                        buffer[full]->blockColPtr + (2*i+2)*(n+1),   // Z colptr
                        buffer[full]->rowIdx + (2*i+2)*CLEN,         // Z rowidx
                        buffer[full]->val + (2*i+2)*CLEN,            // Z val
                        buffer[1-full]->blockColPtr + i*(n+1),       // C colptr
                        buffer[1-full]->rowIdx + 2*i*CLEN,           // C rowidx
                        buffer[1-full]->val + 2*i*CLEN               // C val
                    );
                } else {
                    DPRINT_MSG("Join %d + %d", 2*i, 2*i+1)
                    merge_pcsc_2( 
                        n, nnz,
                        buffer[full]->blockColPtr + 2*i*(n+1),       // A colptr
                        buffer[full]->rowIdx + 2*i*CLEN,             // A rowidx
                        buffer[full]->val + 2*i*CLEN,                // A val
                        buffer[full]->blockColPtr + (2*i+1)*(n+1),   // B colptr
                        buffer[full]->rowIdx + (2*i+1)*CLEN,         // B rowidx
                        buffer[full]->val + (2*i+1)*CLEN,            // B val
                        buffer[1-full]->blockColPtr + i*(n+1),       // C colptr
                        buffer[1-full]->rowIdx + 2*i*CLEN,           // C rowidx
                        buffer[1-full]->val + 2*i*CLEN               // C val
                    );
                }
                DPRINT_MSG("Row idx:")
                DPRINT_ARR(buffer[1-full]->rowIdx, nnz)
            }

            CLEN *= 2;
            full = 1 - full; // ping pong buffer
        }

        std::memcpy(cscColPtr, buffer[full]->blockColPtr, (n+1)*sizeof(int));
        std::memcpy(cscRowIdx, buffer[full]->rowIdx,      nnz*sizeof(int));
        std::memcpy(cscVal,    buffer[full]->val,         nnz*sizeof(int));

        delete colIdxSorted;
        delete buffer[0], buffer[1];
        return COMPUTATION_OK;
    }*/
    
}


#endif