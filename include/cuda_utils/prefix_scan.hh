// Source: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
// and: https://raw.githubusercontent.com/mattdean1/cuda/master/parallel-scan/Submission.cu

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>

// scan.cuh
void scan_on_cuda(int *output, int *input, int length, bool bcao);


float scan(int *output, int *input, int length, bool bcao);
long sequential_scan(int* output, int* input, int length);
float blockscan(int *output, int *input, int length, bool bcao);

void scanLargeDeviceArray(int *output, int *input, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *output, int *input, int length, bool bcao);


// kernels.cuh
__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo);
__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo);

__global__ void prescan_large(int *output, int *input, int n, int* sums);
__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums);

__global__ void add(int *output, int length, int *n1);
__global__ void add(int *output, int length, int *n1, int *n2);


// utils.h
void _checkCudaError(const char *message, cudaError_t err, const char *caller);
void printResult(const char* prefix, int result, long nanoseconds);
void printResult(const char* prefix, int result, float milliseconds);

bool isPowerOfTwo(int x);
int nextPowerOfTwo(int x);

long get_nanos();