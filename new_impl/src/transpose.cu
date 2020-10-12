#include "procedures.hh"

#define BLOCK_DIM 16

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void transpose(int *odata, int *idata, int width, int height)
{
	__shared__ int block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

    // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

void procedures::cuda::transpose(int INPUT_ARRAY input, int * output, int width, int height) {

    if(width % BLOCK_DIM || height % BLOCK_DIM) {
        std::cerr << "Size is not correct\n";
        return;
    }

    dim3 grid(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    transpose<<< grid, threads >>>(input, output, width, height);
    CUDA_CHECK_ERROR
}

void procedures::reference::transpose(int INPUT_ARRAY input, int * output, int width, int height) {
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            input[i * width + j] = output[j * width + i];
        }
    }
}
