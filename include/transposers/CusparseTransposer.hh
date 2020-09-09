#pragma once

#include "CudaTransposer.hh"
#include "cublas_v2.h"
#include "cusparse_v2.h"

class CusparseTransposer : public CudaTransposer {

protected:

    const char* _cusparseGetErrorName(int status) {
        switch(status) {
            case CUSPARSE_STATUS_SUCCESS	        : return "CUSPARSE_STATUS_SUCCESS: the operation completed successfully.";
            case CUSPARSE_STATUS_NOT_INITIALIZED	: return "CUSPARSE_STATUS_NOT_INITIALIZED: the library was not initialized.";
            case CUSPARSE_STATUS_ALLOC_FAILED	    : return "CUSPARSE_STATUS_ALLOC_FAILED: the reduction buffer could not be allocated.";
            case CUSPARSE_STATUS_INVALID_VALUE	    : return "CUSPARSE_STATUS_INVALID_VALUE: the idxBase is neither CUSPARSE_INDEX_BASE_ZERO nor CUSPARSE_INDEX_BASE_ONE.";
            case CUSPARSE_STATUS_ARCH_MISMATCH	    : return "CUSPARSE_STATUS_ARCH_MISMATCH: the device does not support double precision.";
            case CUSPARSE_STATUS_EXECUTION_FAILED	: return "CUSPARSE_STATUS_EXECUTION_FAILED: the function failed to launch on the GPU.";
            case CUSPARSE_STATUS_INTERNAL_ERROR	    : return "CUSPARSE_STATUS_INTERNAL_ERROR: an internal operation failed (check if you are compiling correctly wrt your GPU architecture).";
            default                                 : return "UNKNOWN ERROR";
        }
    }

    int csr2csc_gpumemory(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal,
        int* csrRowPtr_host, int* csrColIdx_host, float* csrVal_host, 
        int* cscColPtr_host, int* cscRowIdx_host, float* cscVal_host) {

        #if (CUDART_VERSION >= 9000) && (CUDART_VERSION < 10000)

            cusparseHandle_t handle;
            cusparseStatus_t status;

            // 1. allocate resources
            status = cusparseCreate(&handle);
            if(status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "csr2csc_cusparse - Error while calling cusparseCreate: " << _cusparseGetErrorName(status) << std::endl;
                return COMPUTATION_ERROR;
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
                return COMPUTATION_ERROR;
            }

            status = cusparseDestroy(handle);
            if(status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "csr2csc_cusparse - Error while calling cusparseDestroy: " << _cusparseGetErrorName(status) << std::endl;
                return COMPUTATION_ERROR;
            }

            return COMPUTATION_OK;

        #elif (CUDART_VERSION >= 10000) 

            cusparseHandle_t handle;
            cusparseStatus_t status;
            size_t buffer_size;

            // 1. allocate resources
            status = cusparseCreate(&handle);
            if(status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "csr2csc_cusparse - Error while calling cusparseCreate: " << cusparseGetErrorName(status) << std::endl;
                return COMPUTATION_ERROR;
            }

            // 2. ask cusparse how much space it needs to operate
            status = cusparseCsr2cscEx2_bufferSize(
                handle,                     // link to cusparse engine
                m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscColPtr, cscRowIdx, 
                CUDA_R_32F,                 // [valType] data type of csrVal, cscVal arrays is 32-bit real (non-complex) single precision floating-point
                CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
                CUSPARSE_INDEX_BASE_ZERO,   // [idxBase]
                CUSPARSE_CSR2CSC_ALG1,      // which algorithm is used? CUSPARSE_CSR2CSC_ALG1 or CUSPARSE_CSR2CSC_ALG2
                &buffer_size);              // fill buffer_size variable

            if(status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2_bufferSize: " << cusparseGetErrorName(status) << std::endl;
                cusparseDestroy(handle);
                return COMPUTATION_ERROR;
            } else if(buffer_size <= 0) {
                std::cerr << "csr2csc_cusparse - warning: buffer_size is not positive" << std::endl;
            }

            // 3. callocate buffer space
            void* buffer = NULL;
            SAFE_CALL( cudaMalloc(&buffer, buffer_size) );
            std::cout << "Needed " << buffer_size << " bytes to esecute Csr2csc" << std::endl; 

            // 4. call transpose
            status = cusparseCsr2cscEx2(
                handle, 
                m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscColPtr, cscRowIdx, 
                CUDA_R_32F,                 // [valType] data type of csrVal, cscVal arrays is 32-bit real (non-complex) single precision floating-point
                CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
                CUSPARSE_INDEX_BASE_ZERO,   // [idxBase]
                CUSPARSE_CSR2CSC_ALG1,      // which algorithm is used? CUSPARSE_CSR2CSC_ALG1 or CUSPARSE_CSR2CSC_ALG2
                buffer);                    // cuda buffer

            if(status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2: " << cusparseGetErrorName(status) << std::endl;
                std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2: " << cusparseGetErrorString(status) << std::endl;
                SAFE_CALL( cudaFree( buffer ) );
                cusparseDestroy(handle);
                return COMPUTATION_ERROR;
            }

            SAFE_CALL( cudaFree( buffer ) );
            status = cusparseDestroy(handle);
            if(status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "csr2csc_cusparse - Error while calling cusparseDestroy: " << cusparseGetErrorName(status) << std::endl;
                return COMPUTATION_ERROR;
            }

            return COMPUTATION_OK;
            
        #else

            #error "Versione di CUDA non supportata"

        #endif
    }

public:

    CusparseTransposer() : CudaTransposer() { }

};