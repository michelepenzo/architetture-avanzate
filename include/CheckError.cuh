#pragma once

#define CHECK_CUDA_ERROR                                                       \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        /*cuda_error::getLastCudaError(__FILE__, __LINE__, __func__);*/        \
    }

#define SAFE_CALL(function)                                                    \
    {                                                                          \
        cuda_error::safe_call(function, __FILE__, __LINE__, __func__);         \
    }

//------------------------------------------------------------------------------

namespace cuda_error {

//inline void getLastCudaError(const char* file, int line, const char* func_name);

inline void safe_call(cudaError_t error,
               const char* file,
               int         line,
               const char* func_name);

inline void cudaErrorHandler(cudaError_t error,
                      const char* error_message,
                      const char* file,
                      int         line,
                      const char* func_name);

} // namespace cuda_error

#include "CheckError.i.cuh"
