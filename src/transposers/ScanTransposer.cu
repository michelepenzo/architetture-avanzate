#include "transposers/ScanTransposer.hh"

__global__ 
void scan_trans_kernel(
    int m, int n, int nnz, 
    int *csrRowPtr, int *csrColIdx, float *csrVal,
    int *cscColPtr, int *cscRowIdx, float *cscVal, 
    int *inter, int *intra)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // intra e inter
    if (global_id < nnz)
    {
        int index = (global_id + 1) * n + csrColIdx[global_id];
        intra[global_id] = inter[index];
        inter[index] = inter[index] + 1;
    }
}

int ScanTransposer::csr2csc_gpumemory(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, float *csrVal, int *cscColPtr, int *cscRowIdx, float *cscVal)
{

    const int N = nnz;
    const int BLOCK_SIZE_X = 256;
    int *intra, *inter, *intra_host, *inter_host;

    // resource allocation
    intra_host = new int[nnz]();
    inter_host = new int[(N + 1) * n]();
    cudaMalloc(&intra, (nnz) * sizeof(int));
    cudaMalloc(&inter, ((N + 1) * n) * sizeof(int));

    // pass info
    SAFE_CALL(cudaMemcpy(intra, intra_host, (nnz) * sizeof(int),         cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(inter, inter_host, ((N + 1) * n) * sizeof(int), cudaMemcpyHostToDevice));

    // kernel execution
    dim3 DimGrid(N / BLOCK_SIZE_X, 1, 1);
    if (N % BLOCK_SIZE_X)
        DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE_X, 1, 1);

    scan_trans_kernel<<<DimGrid, DimBlock>>>(
        m, n, nnz,
        csrRowPtr, csrColIdx, csrVal,
        cscColPtr, cscRowIdx, cscVal,
        inter, intra);
    CHECK_CUDA_ERROR

    // retrieve info
    SAFE_CALL(cudaMemcpy(intra_host, intra, (nnz) * sizeof(int),         cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(inter_host, inter, ((N + 1) * n) * sizeof(int), cudaMemcpyDeviceToHost));

    // debug prints
    std::cout << "Intra: " << std::endl;
    for (int i = 0; i < nnz; i++)
    {
        std::cout << intra_host[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Inter: ";
    for (int i = 0; i < (N + 1); i++)
    {
        std::cout << std::endl << "Row " << i - 1 << ": ";
        for (int j = 0; j < n; j++)
        {
            std::cout << inter_host[i * n + j] << " ";
        }
    }
    std::cout << std::endl;

    // resource deallocation
    delete[] intra_host;
    delete[] inter_host;
    cudaFree(intra);
    cudaFree(inter);

    return COMPUTATION_ERROR;
}