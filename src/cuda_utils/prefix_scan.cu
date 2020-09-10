#include "cuda_utils/prefix_scan.hh"
#include "transposers/CudaTransposer.hh"
#include <iostream>
#include <iomanip>

int THREADS_PER_BLOCK = 8;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

long sequential_scan(int* output, int* input, int length) {
	long start_time = get_nanos();

	output[0] = 0; // since this is a prescan, not a scan
	for (int j = 1; j < length; ++j)
	{
		output[j] = input[j - 1] + output[j - 1];
	}

	long end_time = get_nanos();
	return end_time - start_time;
}

float blockscan(int *output, int *input, int length, bool bcao) {
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int powerOfTwo = nextPowerOfTwo(length);
	if (bcao) {
		prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	}

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

float scan(int *output, int *input, int length, bool bcao) {
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

	// start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	if (length > ELEMENTS_PER_BLOCK) {
		scanLargeDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		scanSmallDeviceArray(d_out, d_in, length, bcao);
	}

	// end timer
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}

void scan_on_cuda(int *d_out, int *d_in, int length, bool bcao) {

	int *d_in_host  = new int[length];
	int *d_out_host = new int[length];

	SAFE_CALL(cudaMemcpy( d_in_host,  d_in, length*sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "\n\n### scan_on_cuda: d_in_host  : ";
	for(int i = 0; i < length; i++) {
        std::cout << std::setw(2) << d_in_host[i] << " ";
    }
	std::cout << " (" << d_in_host[length] << ")" << "\n";
	
	if (length > ELEMENTS_PER_BLOCK) {
		std::cout << "scan_on_cuda: length=" << length << " chooses scanLargeDeviceArray\n";
		scanLargeDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		std::cout << "scan_on_cuda: length=" << length << " chooses scanSmallDeviceArray\n";
		scanSmallDeviceArray(d_out, d_in, length, bcao);
		std::cout << "scan_on_cuda: ended scanSmallDeviceArray" << std::endl;
	}

	SAFE_CALL(cudaMemcpy( d_in_host,  d_in, length*sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "\n\n### scan_on_cuda: d_in_host  : ";
	for(int i = 0; i < length; i++) {
        std::cout << std::setw(2) << d_in_host[i] << " ";
    }
	std::cout << " (" << d_in_host[length] << ")" << "\n";

	SAFE_CALL(cudaMemcpy(d_out_host, d_out, length*sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "### scan_on_cuda: d_out_host : ";
	for(int i = 0; i < length; i++) {
        std::cout << std::setw(2) << d_out_host[i] << " ";
    }
	std::cout << " (" << d_out_host[length] << ")" << "\n";

	delete d_in_host;
	delete d_out_host;
	
}

__global__ void my_add(int *array, int len) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < len) {

		const int OFFSET = *(array - 1);

		array[i] = array[i] + OFFSET;
	}
}

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	std::cout << "scanLargeDeviceArray: remainder " << remainder << "\n";
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		std::cout << "scanLargeDeviceArray: **** 16 elem ARRAY **** from=0 to=" << lengthMultiple << "\n";
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		std::cout << "scanLargeDeviceArray: **** 1 elem ARRAY **** from=" << lengthMultiple << " len=" << remainder << "\n";
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

		std::cout << "scanLargeDeviceArray: add\n";

		//const int OFFSET = d_out[lengthMultiple-1]; // ultimo elemento processato da `scanLargeEvenDeviceArray`

		// a tutti gli elementi di d_out[lengthMultiple...length] aggiungo OFFSET
		//for(int j = lengthMultiple; j < length; j++) {
		//	d_out[j] += OFFSET;
		//}

		my_add<<<1, remainder>>>(startOfOutputArray, remainder);

		//add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	
		
		/*__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}*/
	
	
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao) {
		prescan_arbitrary<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized<<<1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>>>(d_out, d_in, length, powerOfTwo);
	}
	
	/*std::cout << "\t scanSmallDeviceArray: length=" << length << "; powerOfTwo=" << powerOfTwo << "\n";

	int *d_out_host = new int[length];
	int *d_in_host  = new int[length];
	//SAFE_CALL(cudaMemcpy(d_out_host, d_out, length*sizeof(int), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy( d_in_host,  d_in, length*sizeof(int), cudaMemcpyDeviceToHost));
	
	std::cout << "d_in_host : ";
	for(int i = 0; i < length; i++) {
        std::cout << std::setw(2) << d_in_host[i] << " ";
    }
	std::cout << "\n";
	
	int n = length;	


	// copio `d_in_host` dentro `d_out_host`
	for(int i = 0; i < length; i++) {
		const int temp = d_in_host[i];
        d_out_host[i] = temp;
    }

	// applico prefix sum "in-place" che so che funziona (?)
	for(int i = 0; i < n; i++) {
        d_out_host[i+1] += d_out_host[i];
    }
    for(int i = n-1; i >= 0; i--) {
        d_out_host[i+1] = d_out_host[i];
    }
    d_out_host[0] = 0;



	std::cout << "d_out_host: ";
	for(int i = 0; i < length; i++) {
        std::cout << std::setw(2) << d_out_host[i] << " ";
    }
	std::cout << "\n";

	SAFE_CALL(cudaMemcpy(d_out, d_out_host, length*sizeof(int), cudaMemcpyHostToDevice));
    //SAFE_CALL(cudaMemcpy( d_in,  d_in_host, length*sizeof(int), cudaMemcpyHostToDevice));
    
	delete d_out_host;
	delete d_in_host; */
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {

	std::cout << "\t scanLargeEvenDeviceArray: length " << length << "\n";
	
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	if (bcao) {
		prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5

// There were two BCAO optimisations in the paper - this one is fastest
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);


	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
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


__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
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


__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

// from https://stackoverflow.com/a/3638454
bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}

// from https://stackoverflow.com/a/12506181
int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}


// from https://stackoverflow.com/a/36095407
// Get the current time in nanoseconds
long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}
