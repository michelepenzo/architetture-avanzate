#include "transposers.hh"

void transposers::serial_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {
    
    int* curr = new int[n+1](); // array inizializzato con tutti '0'

    DPRINT_MSG("Reference")
    DPRINT_ARR(csrRowPtr, m+1)
    DPRINT_ARR(csrColIdx, nnz)
    DPRINT_ARR(csrVal, nnz)

    // 1. costruisco `cscColPtr` come istogramma delle frequenze degli elementi per ogni colonna
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            int index = csrColIdx[j];
            cscColPtr[index]++;
        }
    }

    // 2. applico prefix_sum per costruire corretto `cscColPtr` (ogni cella tiene conto dei precedenti)
    utils::prefix_sum(cscColPtr, n+1);

    // 3. sistemo indici di riga e valori
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            int col = csrColIdx[j];
            int loc = cscColPtr[col] + curr[col];
            if(loc > nnz) {
                printf("ERROR i=%d j=%d from %d to %d\n", i, j, csrRowPtr[i], csrRowPtr[i+1]);
                printf("loc=%d+%d=%d>%d=NNZ\n", cscColPtr[col], curr[col], loc, nnz);
            }
            curr[col]++;
            cscRowIdx[loc] = i;
            cscVal[loc] = csrVal[j];
        }
    }

    DPRINT_MSG("SERIAL")
    DPRINT_ARR(cscColPtr, n+1)
    DPRINT_ARR(cscRowIdx, nnz)
    DPRINT_ARR(   cscVal, nnz)

    delete[] curr;
}

void transposers::cuda_wrapper(
    int m, int n, int nnz,
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal,
    algo _algo
) {

    int * csrRowPtr_cuda = utils::cuda::allocate_send<int>(csrRowPtr, m+1);
    int * csrColIdx_cuda = utils::cuda::allocate_send<int>(csrColIdx, nnz);
    float * csrVal_cuda  = utils::cuda::allocate_send<float>(csrVal, nnz);

    int * cscColPtr_cuda = utils::cuda::allocate_zero<int>(n+1);
    int * cscRowIdx_cuda = utils::cuda::allocate_zero<int>(nnz);
    float * cscVal_cuda  = utils::cuda::allocate_zero<float>(nnz);

    _algo(
        m, n, nnz,
        csrRowPtr_cuda, csrColIdx_cuda, csrVal_cuda, 
        cscColPtr_cuda, cscRowIdx_cuda, cscVal_cuda
    );

    utils::cuda::deallocate(csrRowPtr_cuda);
    utils::cuda::deallocate(csrColIdx_cuda);
    utils::cuda::deallocate(csrVal_cuda);

    utils::cuda::deallocate_recv<int>(cscColPtr, cscColPtr_cuda, n+1);
    utils::cuda::deallocate_recv<int>(cscRowIdx, cscRowIdx_cuda, nnz);
    utils::cuda::deallocate_recv<float>(cscVal, cscVal_cuda, nnz);
}

__global__ 
void reorder_elements_kernel(
    int n, int nnz,
    int *inter, int *intra, 
    int *csrRowIdx, int *csrColIdx, float *csrVal,
    int *cscColPtr, int *cscRowIdx, float *cscVal
) {

    const int j = blockIdx.x;
    
    // allineo inter alla riga corretta
    inter += j*n;

    // recupero gli estremi della porzione di array da processare
    const int BLOCK_SIZE = DIV_THEN_CEIL(nnz, HISTOGRAM_BLOCKS);
    const int start = j*BLOCK_SIZE;
    const int end = min((j+1)*BLOCK_SIZE, nnz);
    const int len = end - start;

    // calcolo la posizione degli elementi
    for(int i = 0; i < len; i++) {
        int cid = csrColIdx[start + i];
        int loc = cscColPtr[cid] + inter[cid] + intra[start + i];
        if(loc > nnz) {
            //printf("(%2d): START=%d, END=%d, LEN=%d\n", j, start, end, len);
            //printf("start+i=%d\n", start+i);
            //printf("ERROR loc=%d+%d+%d = %d > %d=NNZ\n", 
            //cscColPtr[cid], inter[cid], intra[start + i], loc, nnz);
        } else {
            cscRowIdx[loc] = csrRowIdx[start + i];
            cscVal[loc] = csrVal[start + i];
        }
    }
}

void transposers::scan_csr2csc(
    int m, int n, int nnz,
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    // 1. espandi l'array di puntatori agli indici di riga, nell'array di indici esteso
    int * csrRowIdx = utils::cuda::allocate_zero<int>(nnz);
    procedures::cuda::pointers_to_indexes(csrRowPtr, m, csrRowIdx, nnz);
    DPRINT_ARR_CUDA(csrRowIdx, nnz)

    // 2. riempi inter, intra, e colPtr
    int * inter;
    int * intra = utils::cuda::allocate_zero<int>(nnz);
    int * colPtr = utils::cuda::allocate_zero<int>(n+1);
    procedures::cuda::indexes_to_pointers(csrColIdx, nnz, &inter, intra, colPtr, n);
    for(int i = 0; i <= HISTOGRAM_BLOCKS; i++) {
        DPRINT_ARR_CUDA(inter+i*n, n)
    }
    DPRINT_ARR_CUDA(intra, nnz)

    // 3. applica scan ai puntatori
    procedures::cuda::scan(colPtr, cscColPtr, n+1);

    // 4. permuta valori
    reorder_elements_kernel<<<HISTOGRAM_BLOCKS, 1>>>(
        n, nnz,
        inter, intra, 
        csrRowIdx, csrColIdx, csrVal, 
        cscColPtr, cscRowIdx, cscVal);
    CUDA_CHECK_ERROR

    DPRINT_MSG("SCAN")
    DPRINT_ARR_CUDA(cscColPtr, n+1)
    DPRINT_ARR_CUDA(cscRowIdx, nnz)
    DPRINT_ARR_CUDA(cscVal, nnz)
    for(int i = 0; i <= HISTOGRAM_BLOCKS; i++) {
        DPRINT_ARR_CUDA(inter+i*n, n)
    }
    DPRINT_ARR_CUDA(intra, nnz)

    utils::cuda::deallocate(csrRowIdx);
    utils::cuda::deallocate(inter);
    utils::cuda::deallocate(intra);
    utils::cuda::deallocate(colPtr);
}

__global__ 
void copy_histo_kernel(
    int * colPtrIn,  int * rowIdxIn,  float * valIn,
    int * colPtrOut, int * rowIdxOut, float * valOut,
    int n, int nnz, int BLOCK_SIZE
) {
    int i = blockIdx.x;

    int * colPtrA = colPtrIn + 2*i*(n+1);
    int * colPtrB = colPtrIn + (2*i+1)*(n+1);
    int * colPtrC = colPtrOut + 2*i*(n+1);

    // sistema i puntatori al blocco di uscita
    for(int i = 0; i < n+1; i++) {
        colPtrC[i] = colPtrA[i] + colPtrB[i];
    }

    int * rowIdxA = rowIdxIn + 2*i*BLOCK_SIZE;
    int * rowIdxB = rowIdxIn + (2*i+1)*BLOCK_SIZE;
    int * rowIdxC = rowIdxOut + 2*i*BLOCK_SIZE;

    float * valA = valIn + 2*i*BLOCK_SIZE;
    float * valB = valIn + (2*i+1)*BLOCK_SIZE;
    float * valC = valOut + 2*i*BLOCK_SIZE;

    // copia in output i valori corretti di rowIdx e val
    for(int i = 0; i < n; i++) {

        int sa = colPtrA[i], la = colPtrA[i+1] - sa;
        int sb = colPtrB[i], lb = colPtrB[i+1] - sb;
        int sc = colPtrC[i];//, lc = colPtrC[i+1] - sc;

        //DPRINT_MSG("Col=%d, sa=%d, sb=%d, sc=%d, la=%d, lb=%d, lc=%d", i, sa, sb, sc, la, lb, lc)

        utils::cuda::devcopy<int>(rowIdxC+sc, rowIdxA+sa, la);
        utils::cuda::devcopy<float>( valC+sc,    valA+sa, la);

        sc = sc + la;
        utils::cuda::devcopy<int>(rowIdxC+sc, rowIdxB+sb, lb);
        utils::cuda::devcopy<float>( valC+sc,    valB+sb, lb);
    }
}

void transposers::merge_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal) {

    DPRINT_ARR_CUDA(csrRowPtr, m+1);
    DPRINT_ARR_CUDA(csrColIdx, nnz);
    DPRINT_ARR_CUDA(csrVal,    nnz);

    // alloca lo spazio necessario per effettuare il sort
    struct merge_buffer {
        //int * colPtrBlock;
        int * colIdx;
        int * rowIdx;
        float * val;
    };
    merge_buffer buffer[2];
    buffer[0].colIdx = utils::cuda::allocate_zero<int>(nnz);
    buffer[0].rowIdx = utils::cuda::allocate_zero<int>(nnz);
    buffer[0].val    = utils::cuda::allocate_zero<float>(nnz);
    buffer[1].colIdx = utils::cuda::allocate_zero<int>(nnz);
    buffer[1].rowIdx = cscRowIdx;
    buffer[1].val    = cscVal;
    int * colPtr     = utils::cuda::allocate_zero<int>(n+1);

    // 1. espandi rowPtr in rowIdx
    DPRINT_MSG("1 ---- row IDX to PTR")
    procedures::cuda::pointers_to_indexes(csrRowPtr, m, buffer[0].rowIdx, nnz);
    DPRINT_ARR_CUDA(buffer[0].rowIdx, nnz);

    // 2. ordina per indice delle colonne
    DPRINT_MSG("2 ---- sort by column index")
    utils::cuda::copy(buffer[0].colIdx, csrColIdx, nnz);
    procedures::cuda::segsort3(buffer[0].colIdx, buffer[1].colIdx, nnz, buffer[0].rowIdx, buffer[1].rowIdx, csrVal, buffer[1].val);
    DPRINT_ARR_CUDA(buffer[1].colIdx,  nnz);
    DPRINT_ARR_CUDA(buffer[1].rowIdx, nnz);
    DPRINT_ARR_CUDA(buffer[1].val,    nnz);

    // 3. merging
    DPRINT_MSG("3 ---- merging")
    int full = 1;
    int CURRENT_BLOCK_SIZE = SEGSORT_ELEMENTS_PER_BLOCK;
    int TARGET_BLOCK_SIZE = (nnz-1)*2;

    DPRINT_MSG("INIT Block Size %d", CURRENT_BLOCK_SIZE)
    DPRINT_ARR_CUDA(buffer[full].colIdx,  nnz);
    DPRINT_ARR_CUDA(buffer[full].rowIdx, nnz);
    DPRINT_ARR_CUDA(buffer[full].val,    nnz);

    while(CURRENT_BLOCK_SIZE < TARGET_BLOCK_SIZE) {
    
        procedures::cuda::merge3_step<int, int, float>(
            buffer[full].colIdx, buffer[1-full].colIdx, 
            buffer[full].rowIdx, buffer[1-full].rowIdx, 
            buffer[full].val, buffer[1-full].val,
            nnz, CURRENT_BLOCK_SIZE
        );
    
        full = 1 - full;
        CURRENT_BLOCK_SIZE *= 2;
        DPRINT_MSG("Block size %d", CURRENT_BLOCK_SIZE)
        DPRINT_ARR_CUDA(buffer[full].colIdx,  nnz);
        DPRINT_ARR_CUDA(buffer[full].rowIdx, nnz);
        DPRINT_ARR_CUDA(buffer[full].val,    nnz);
    }

    // 4. index to pointers
    DPRINT_MSG("4 ---- colIdx -> colPtr")
    int * inter;
    procedures::cuda::indexes_to_pointers(buffer[full].colIdx, nnz, &inter, colPtr, n);
    procedures::cuda::scan(colPtr, cscColPtr, n+1);

    if(full != 1) {
        utils::cuda::copy<int>(cscRowIdx, buffer[full].rowIdx, nnz);
        utils::cuda::copy<float>(cscVal, buffer[full].val, nnz);
    }

    DPRINT_ARR_CUDA(cscColPtr, n+1);
    DPRINT_ARR_CUDA(cscRowIdx, nnz);
    DPRINT_ARR_CUDA(cscVal,    nnz);

    utils::cuda::deallocate(inter);
    utils::cuda::deallocate(colPtr);
    utils::cuda::deallocate(buffer[0].colIdx);
    utils::cuda::deallocate(buffer[1].colIdx);
    utils::cuda::deallocate(buffer[0].rowIdx);
    utils::cuda::deallocate(buffer[0].val);
}

const char* _cusparseGetErrorName(int status) {
    switch(status) {
        case CUSPARSE_STATUS_SUCCESS            : return "CUSPARSE_STATUS_SUCCESS: the operation completed successfully.";
        case CUSPARSE_STATUS_NOT_INITIALIZED    : return "CUSPARSE_STATUS_NOT_INITIALIZED: the library was not initialized.";
        case CUSPARSE_STATUS_ALLOC_FAILED       : return "CUSPARSE_STATUS_ALLOC_FAILED: the reduction buffer could not be allocated.";
        case CUSPARSE_STATUS_INVALID_VALUE      : return "CUSPARSE_STATUS_INVALID_VALUE: the idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.";
        case CUSPARSE_STATUS_ARCH_MISMATCH      : return "CUSPARSE_STATUS_ARCH_MISMATCH: the device does not support double precision.";
        case CUSPARSE_STATUS_EXECUTION_FAILED   : return "CUSPARSE_STATUS_EXECUTION_FAILED: the function failed to launch on the GPU.";
        case CUSPARSE_STATUS_INTERNAL_ERROR     : return "CUSPARSE_STATUS_INTERNAL_ERROR: an internal operation failed (check if you are compiling correctly wrt your GPU architecture).";
        default                                 : return "UNKNOWN ERROR";
    }
}

void cusparse_generic_csr2csc_gpumemory(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal, bool use_algo1) {

    #if (CUDART_VERSION >= 9000) && (CUDART_VERSION < 10000)

        cusparseHandle_t handle;
        cusparseStatus_t status;

        // 1. allocate resources
        status = cusparseCreate(&handle);
        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseCreate: " << _cusparseGetErrorName(status) << std::endl;
            return;
        }

        // 2. call transpose
        // reference: https://docs.nvidia.com/cuda/archive/9.0/cusparse/index.html#cusparse-lt-t-gt-csr2csc
        status = cusparseScsr2csc(
            handle, 
            m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscRowIdx, cscColPtr,
            CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
            CUSPARSE_INDEX_BASE_ZERO);  // [idxBase]

        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseScsr2csc: " << _cusparseGetErrorName(status) << std::endl;
            cusparseDestroy(handle);
            return;
        }

        status = cusparseDestroy(handle);
        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseDestroy: " << _cusparseGetErrorName(status) << std::endl;
            return;
        }

        return;

    #elif (CUDART_VERSION >= 10000) 

        cusparseHandle_t handle;
        cusparseStatus_t status;
        size_t buffer_size;

        // 1. allocate resources
        status = cusparseCreate(&handle);
        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseCreate: " << cusparseGetErrorName(status) << std::endl;
            return;
        }

        // 2. ask cusparse how much space it needs to operate
        status = cusparseCsr2cscEx2_bufferSize(
            handle,                     // link to cusparse engine
            m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscColPtr, cscRowIdx, 
            CUDA_R_32F,                 // [valType] data type of csrVal, cscVal arrays is 32-bit real (non-complex) single precision floating-point
            CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
            CUSPARSE_INDEX_BASE_ZERO,   // [idxBase]
            (use_algo1 ? CUSPARSE_CSR2CSC_ALG1 : CUSPARSE_CSR2CSC_ALG2),
                                        // which algorithm is used? CUSPARSE_CSR2CSC_ALG1 or CUSPARSE_CSR2CSC_ALG2
            &buffer_size);              // fill buffer_size variable

        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2_bufferSize: " << cusparseGetErrorName(status) << std::endl;
            cusparseDestroy(handle);
            return;
        } else if(buffer_size <= 0) {
            std::cerr << "csr2csc_cusparse - warning: buffer_size is not positive" << std::endl;
        }

        // 3. callocate buffer space
        void* buffer = NULL;
        CUDA_SAFE_CALL( cudaMalloc(&buffer, buffer_size) );
        //std::cout << "Needed " << buffer_size << " bytes to esecute Csr2csc" << std::endl; 

        // 4. call transpose
        status = cusparseCsr2cscEx2(
            handle, 
            m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscColPtr, cscRowIdx, 
            CUDA_R_32F,                 // [valType] data type of csrVal, cscVal arrays is 32-bit real (non-complex) single precision floating-point
            CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
            CUSPARSE_INDEX_BASE_ZERO,   // [idxBase]
            (use_algo1 ? CUSPARSE_CSR2CSC_ALG1 : CUSPARSE_CSR2CSC_ALG2),
                                        // which algorithm is used? CUSPARSE_CSR2CSC_ALG1 or CUSPARSE_CSR2CSC_ALG2
            buffer);                    // cuda buffer

        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2: " << cusparseGetErrorName(status) << std::endl;
            std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2: " << cusparseGetErrorString(status) << std::endl;
            CUDA_SAFE_CALL( cudaFree( buffer ) );
            cusparseDestroy(handle);
            return;
        }

        CUDA_SAFE_CALL( cudaFree( buffer ) );
        status = cusparseDestroy(handle);
        if(status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "csr2csc_cusparse - Error while calling cusparseDestroy: " << cusparseGetErrorName(status) << std::endl;
            return;
        }
        
    #else

        #error "Versione di CUDA non supportata"

    #endif
}

void transposers::cusparse1_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal)
{
    cusparse_generic_csr2csc_gpumemory(
        m, n, nnz, 
        csrRowPtr, csrColIdx, csrVal, 
        cscColPtr, cscRowIdx, cscVal, 
        true
    );
}

void transposers::cusparse2_csr2csc(
    int m, int n, int nnz, 
    int* csrRowPtr, int* csrColIdx, float* csrVal, 
    int* cscColPtr, int* cscRowIdx, float* cscVal)
{
    cusparse_generic_csr2csc_gpumemory(
        m, n, nnz, 
        csrRowPtr, csrColIdx, csrVal, 
        cscColPtr, cscRowIdx, cscVal, 
        false
    );
}