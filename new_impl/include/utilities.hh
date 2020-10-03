#pragma once
#ifndef UTILITIES_HH_
#define UTILITIES_HH_

#include <iostream>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <string>
#include <type_traits>
#include <cassert>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#define INPUT_ARRAY * const
#define NUMERIC_TEMPLATE(T) template< typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type >
#define DIV_THEN_CEIL(a,b) a%b?((a/b)+1):(a/b)

#define COMPUTATION_OK 0
#define COMPUTATION_ERROR -1

#define CUDA_CHECK_ERROR { utils::cuda::check_error(__FILE__, __LINE__, __func__); }
#define CUDA_SAFE_CALL(function) { utils::cuda::safe_call(function, __FILE__, __LINE__, __func__); }

#define DEBUG_MODE 1

#if DEBUG_MODE == 1
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#define ASSERT_LIMIT(value,limit) { utils::assert_limit(value,limit,__FILE__, __LINE__, __func__); }
#define ASSERT_RANGE(value) { utils::assert_range(value, SPARSE_MATRIX_MIN_VAL, SPARSE_MATRIX_MAX_VAL, __FILE__, __LINE__, __func__); }
#define DPRINT_MSG(message, ...) { printf("%20s: ", __func__); printf(message "\n", ##__VA_ARGS__); }
#define DPRINT_ARR(array, len) { printf("%20s: ", __func__); utils::print(STRINGIFY(array), array, len); }
#define DPRINT_ARR_CUDA(array, len) { printf("%20s: ", __func__); utils::cuda::print(STRINGIFY(array), array, len); }
#else 
#define ASSERT_LIMIT(value,limit) { value; limit; }
#define ASSERT_RANGE(value) { value; }
#define DPRINT_MSG(message, ...) { message; } 
#define DPRINT_ARR(array, len) { array; len; }
#define DPRINT_ARR_CUDA(array, len) { array; len; }
#endif

namespace utils {

    NUMERIC_TEMPLATE(T)
    inline bool equals(T INPUT_ARRAY first, T INPUT_ARRAY second, int len) {
        for(int i = 0; i < len; i++) { // metti std::equal
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

    NUMERIC_TEMPLATE(T)
    inline void copy_array(T * dest, T INPUT_ARRAY src, int len) {
        std::memcpy(dest, src, len*sizeof(T));
    }

    NUMERIC_TEMPLATE(T)
    inline T* copy_array(T INPUT_ARRAY array, int len) {
        T* copy = new T[len];
        copy_array<T>(copy, array, len);
        return copy;
    }

    inline int next_two_pow(int number) {
        unsigned int v = (unsigned int) number;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return (int)v;
    }

    namespace random {

        inline std::default_random_engine& generator() {
            static std::default_random_engine g(1234567);
            return g;
        }

        inline void reset_generator(int seed = 0) {
            if(seed == 0) {
                seed = std::chrono::system_clock::now().time_since_epoch().count();
            }
            generator() = std::default_random_engine(seed);
        }

        inline int generate(int min, int max) {
            std::uniform_int_distribution<int> values_distrib(min, max);
            return values_distrib(generator());
        }

        inline int generate(int max) {
            return generate(0, max);
        }

        inline int* generate_array(int min, int max, int len) {
            int* array = new int[len];
            for(int i = 0; i < len; i++) {
                array[i] = generate(min, max);
            }
            return array;
        }
    }

    namespace cuda {

        inline void exit() {
            assert(false); //NOLINT
            std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));
            std::exit(EXIT_FAILURE);
        }

        inline void safe_call(cudaError_t error, const char* file, int line, const char* func) {
            if (cudaSuccess != error) {
                std::cerr << "\nCUDA ERROR " << static_cast<int>(error) << " " << cudaGetErrorString(error) << "\n";
                std::cerr << "In file=" << file << " line=" << line << " func=" << func << "\n";
                exit();
            }
        }

        inline void check_error(const char* file, int line, const char* func_name) {
            cudaDeviceSynchronize(); // wait until kernel stops
            safe_call(cudaGetLastError(), file, line, func_name); // check any error
        }

        NUMERIC_TEMPLATE(T)
        inline T* allocate(int n_elem) {
            T *ptr; 
            CUDA_SAFE_CALL(cudaMalloc(&ptr, n_elem * sizeof(T)));
            return ptr;
        }

        NUMERIC_TEMPLATE(T)
        inline T* allocate_zero(int n_elem) {
            T *ptr = allocate<T>(n_elem);
            CUDA_SAFE_CALL(cudaMemset(ptr, 0, n_elem * sizeof(T)));
            return ptr;
        }

        NUMERIC_TEMPLATE(T)
        inline void send(T * dest_cuda_array, T INPUT_ARRAY src_host_array, int len) {
            CUDA_SAFE_CALL(cudaMemcpy(dest_cuda_array, src_host_array, len*sizeof(T), cudaMemcpyHostToDevice));
        }

        NUMERIC_TEMPLATE(T)
        inline void recv(T * dest_host_array, T INPUT_ARRAY src_cuda_array, int len) {
            CUDA_SAFE_CALL(cudaMemcpy(dest_host_array, src_cuda_array, len*sizeof(T), cudaMemcpyDeviceToHost));
        }

        NUMERIC_TEMPLATE(T)
        inline void copy(T * dest_cuda_array, T INPUT_ARRAY src_cuda_array, int len) {
            CUDA_SAFE_CALL(cudaMemcpy(dest_cuda_array, src_cuda_array, len*sizeof(T), cudaMemcpyDeviceToDevice));
        }

        NUMERIC_TEMPLATE(T)
        inline void deallocate(T *ptr) {
            CUDA_SAFE_CALL(cudaFree(ptr))
        }

        NUMERIC_TEMPLATE(T)
        inline T* allocate_send(T INPUT_ARRAY src_host, int len) {
            T *cuda_array = allocate<T>(len);
            send(cuda_array, src_host, len);
            return cuda_array;
        }

        NUMERIC_TEMPLATE(T)
        inline void deallocate_recv(T * dest_host_array, T INPUT_ARRAY src_cuda_array, int len) {
            recv(dest_host_array, src_cuda_array, len);
            deallocate(src_cuda_array);
        }

        NUMERIC_TEMPLATE(T)
        inline void print(std::string name, T *cuda_array, int len) {
            T *host_array = new int[len];
            recv<T>(host_array, cuda_array, len);
            utils::print(name, host_array, len);
            delete host_array;
        }

    }

    inline void assert_limit(int value, int limit, const char* file, int line, const char* func) {
        if(value > limit) {
            std::cerr << "Limit exceeded of " << value << " (max is " << limit << ")\n";
            std::cerr << "In file=" << file << " line=" << line << " func=" << func << "\n";
            cuda::exit();
        }
    }

    inline void assert_range(float value, float min_val, float max_val, const char* file, int line, const char* func) {
        if(value < min_val || value > max_val) {
            std::cerr << "Limit exceeded of " << value << " (range is " << min_val << " to " << max_val << ")\n";
            std::cerr << "In file=" << file << " line=" << line << " func=" << func << "\n";
            cuda::exit();
        }
    }

}


#endif