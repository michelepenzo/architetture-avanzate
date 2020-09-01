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

    int csr2csc_gpumemory(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal) {

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

        #else

            #error "Versione di CUDA non supportata"

        #endif
    }

public:

    CusparseTransposer(SparseMatrix* sm) : CudaTransposer(sm) { }

};