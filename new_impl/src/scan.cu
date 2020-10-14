#include <iostream>
#include <iomanip>
#include "procedures.hh"

void procedures::reference::scan(int INPUT_ARRAY input, int * output, int len) {
    output[0] = 0;
    for(int i = 1; i < len; i++) {
        output[i] = output[i-1] + input[i-1];
    }
}

static const int THREADS_PER_BLOCK = 512;

static const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void scan_small(int * output, int INPUT_ARRAY input, int length);

void scan_even(int * output, int INPUT_ARRAY input, int length);

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo);

__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums);

void add_multiple_offset(int * output, int BLOCKS, int INPUT_ARRAY incr);

void add_single_offset(int * output, int length, int INPUT_ARRAY n1, int INPUT_ARRAY n2);


void procedures::cuda::scan(int INPUT_ARRAY d_in, int * d_out, int length) {

	if (length <= ELEMENTS_PER_BLOCK) {

		scan_small(d_out, d_in, length);
	}
	else if(length % ELEMENTS_PER_BLOCK == 0) {

		scan_even(d_out, d_in, length);
	} else {

		// perform a large scan on a compatible multiple of elements
		int remainder = length % ELEMENTS_PER_BLOCK;
		int lengthMultiple = length - remainder;
		scan_even(d_out, d_in, lengthMultiple);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scan_small(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

		add_single_offset(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}

	return;
}

void scan_small(int * output, int INPUT_ARRAY input, int length) {
	// non alloco tutta la memoria massima ma solo quella che mi serve
	int LENGHT_TWO_POW = utils::next_two_pow(length);

	// chiamo il kernel che non necessita di somme
    prescan_arbitrary_unoptimized<<<1, DIV_THEN_CEIL(length, 2), 2 * LENGHT_TWO_POW * sizeof(int)>>>(
		output, input, length, LENGHT_TWO_POW);
}

void scan_even(int * output, int INPUT_ARRAY input, int length) {

	// quanti blocchi? la lunghezza è multipla di ELEMENTS_PER_BLOCK
	const int BLOCKS = length / ELEMENTS_PER_BLOCK;
	const int SM_SIZE = 2 * ELEMENTS_PER_BLOCK * sizeof(int);

	// alloco array ausiliari degli offset
	int * sums = utils::cuda::allocate<int>(BLOCKS);
	int * incr = utils::cuda::allocate<int>(BLOCKS);

	// chiamo il kernel (mantengo gli offset parziali in sums)
	prescan_large_unoptimized<<<BLOCKS, THREADS_PER_BLOCK, SM_SIZE>>>(
		output, input, ELEMENTS_PER_BLOCK, sums);

	// applico prefix-scan sugli offset parziali
	procedures::cuda::scan(sums, incr, BLOCKS);

	// sommo gli offset ad output
	add_multiple_offset(output, BLOCKS, incr);

	utils::cuda::deallocate(sums);
	utils::cuda::deallocate(incr);
}

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else { 
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}


__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}

// ==========================================================================
// ADD UTILITY ==============================================================
//==========================================================================

__global__ 
void add(int *output, int INPUT_ARRAY n1, int INPUT_ARRAY n2) {
	int i = threadIdx.x;
	output[i] += n1[0] + n2[0];
}

__global__ 
void add(int * output, int * incr) {
	int j = blockIdx.x;
	int i = threadIdx.x;
	output[j*ELEMENTS_PER_BLOCK + i] += incr[j];
}

void add_single_offset(int * output, int length, int INPUT_ARRAY n1, int INPUT_ARRAY n2) {
	add<<<1, length>>>(output, n1, n2);
}

void add_multiple_offset(int * output, int BLOCKS, int INPUT_ARRAY incr) {
	add<<<BLOCKS, ELEMENTS_PER_BLOCK>>>(output, incr);
}
