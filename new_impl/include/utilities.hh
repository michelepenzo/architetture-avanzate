#pragma once
#include <iostream>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <string>
#include <type_traits>
#include <cassert>
#include <cuda_runtime.h>

#define INPUT_ARRAY const * const
#define NUMERIC_TEMPLATE(T) template< typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type >
#define DIV_THEN_CEIL(a,b) a%b?((a/b)+1):(a/b)

#define SPARSE_MATRIX_MIN_VAL 1
#define SPARSE_MATRIX_MAX_VAL 100
#define COMPUTATION_OK 0
#define COMPUTATION_ERROR -1

#define CUDA_CHECK_ERROR { utils::cuda_check_error(__FILE__, __LINE__, __func__); }
#define CUDA_SAFE_CALL(function) { utils::cuda_safe_call(function, __FILE__, __LINE__, __func__); }

#define DEBUG_MODE 1

#if DEBUG_MODE == 1
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#define ASSERT_LIMIT(value,limit) { utils::assert_limit(value,limit,__FILE__, __LINE__, __func__); }
#define ASSERT_RANGE(value) { utils::assert_range(value, SPARSE_MATRIX_MIN_VAL, SPARSE_MATRIX_MAX_VAL, __FILE__, __LINE__, __func__); }
#define DPRINT_MSG(message, ...) { printf("%20s: ", __func__); printf(message "\n", ##__VA_ARGS__); }
#define DPRINT_ARR(array, len) { printf("%20s: ", __func__); utils::print(STRINGIFY(array), array, len); }
#else 
#define ASSERT_LIMIT(value,limit) { value; limit; }
#define ASSERT_RANGE(value) { value; }
#define DPRINT_MSG(message, ...) { message; } 
#define DPRINT_ARR(array, len) { array; len; }
#endif

namespace utils {

    NUMERIC_TEMPLATE(T)
    inline bool equals(T INPUT_ARRAY first, T INPUT_ARRAY second, int len) {
        for(int i = 0; i < len; i++) {
            if(first[i] != second[i]) {
                return false;
            }
        }
        return true;
    }

    NUMERIC_TEMPLATE(T)
    inline void print(std::string name, T INPUT_ARRAY array, int len) {
        std::cout << std::setw(40) << name << "=";
        for(int i = 0; i < len; i++) {
            std::cout << std::setw(3) << array[i] << " ";
        }
        std::cout << std::endl;
    }

    inline void prefix_sum(int *ptr, int n) {
        for(int j = 1; j < n; j++) {
            ptr[j] += ptr[j-1];
        }
    }

    inline int* create_indexes(int len) {
        int* indices = new int[len];
        for(int i = 0; i < len; i++) indices[i] = i;
        return indices;
    }

    inline int* copy_array(int INPUT_ARRAY array, int len) {
        int* copy = new int[len];
        std::memcpy(copy, array, len*sizeof(int));
        return copy;
    }

    NUMERIC_TEMPLATE(T)
    inline void copy_array(T * dest, T INPUT_ARRAY src, int len) {
        std::memcpy(dest, src, len*sizeof(T));
    }

    void cuda_exit() {
        assert(false); //NOLINT
        std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
        std::exit(EXIT_FAILURE);
    }

    void cuda_safe_call(cudaError_t error, const char* file, int line, const char* func) {
        if (cudaSuccess != error) {
            std::cerr << "\nCUDA ERROR " << static_cast<int>(error) << " " << cudaGetErrorString(error) << "\n";
            std::cerr << "In file=" << file << " line=" << line << " func=" << func << "\n";
            cuda_exit();
        }
    }

    void cuda_check_error(const char* file, int line, const char* func_name) {
        cudaDeviceSynchronize(); // wait until kernel stops
        cuda_safe_call(cudaGetLastError(), file, line, func_name); // check any error
    }

    void assert_limit(int value, int limit, const char* file, int line, const char* func) {
        if(value > limit) {
            std::cerr << "Limit exceeded of " << value << " (max is " << limit << ")\n";
            std::cerr << "In file=" << file << " line=" << line << " func=" << func << "\n";
            cuda_exit();
        }
    }

    void assert_range(float value, float min_val, float max_val, const char* file, int line, const char* func) {
        if(value < min_val || value > max_val) {
            std::cerr << "Limit exceeded of " << value << " (range is " << min_val << " to " << max_val << ")\n";
            std::cerr << "In file=" << file << " line=" << line << " func=" << func << "\n";
            cuda_exit();
        }
    }

}
