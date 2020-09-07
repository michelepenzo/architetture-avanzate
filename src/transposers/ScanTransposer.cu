#include "transposers/ScanTransposer.hh"

#define DIV_THEN_CEIL(a,b) a%b?((a/b)+1):(a/b)

/// =============================================================
/// ============================================ Fill `cscRowIdx`
/// =============================================================

__global__ 
void csrrowidx_kernel(int m, int *csrRowPtr, int *csrRowIdx) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id

    if(i < m) {
        int start = csrRowPtr[i], end = csrRowPtr[i+1];
        for(int j = start; j < end; j++) {
            csrRowIdx[j] = i;
        }
    }
}

void ScanTransposer::csrrowidx_caller(int m, int *csrRowPtr, int *cscRowIdx) {

    dim3 DimGrid(DIV_THEN_CEIL(m, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    csrrowidx_kernel<<<DimGrid, DimBlock>>>(m, csrRowPtr, cscRowIdx);
    CHECK_CUDA_ERROR
}

/// =============================================================
/// ======================================= Fill `intra`, `inter`
/// =============================================================

__global__
void inter_intra_kernel(int n, int nnz, int len, int *inter, int *intra, int *csrColIdx) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    int start = tid * len;

    for(int i = 0; i < len && start + i < nnz; i++) {
        int index = (tid + 1) * n + csrColIdx[start + i];
        intra[start + i] = inter[index];
        inter[index]++;
    }
}

void ScanTransposer::inter_intra_caller(int n, int nnz, int *inter, int *intra, int *csrColIdx) {

    dim3 DimGrid(DIV_THEN_CEIL(N_THREAD, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    const int LEN = DIV_THEN_CEIL(nnz, N_THREAD);

    inter_intra_kernel<<<DimGrid, DimBlock>>>(n, nnz, LEN, inter, intra, csrColIdx);
    CHECK_CUDA_ERROR
}

/// =============================================================
/// ======================================= Apply `vertical_scan`
/// =============================================================

__global__
void vertical_scan_kernel(int n, int nthread, int *inter, int *cscColPtr) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // global_id

    if(i < n) {
        // vertical scan on inter
        for(int j = 0; j < nthread; j++) {
            inter[(j+1)*n+i] += inter[j*n+i]; // inter[j+1][i] += inter[j][i];
        }
        // last element goes into `cscColPtr`
        cscColPtr[i+1] = inter[nthread*n+i]; // cscColPtr[i+1] = inter[nthread][i];
    }
}

void ScanTransposer::vertical_scan_caller(int n, int *inter, int *cscColPtr) {

    dim3 DimGrid(DIV_THEN_CEIL(n, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    vertical_scan_kernel<<<DimGrid, DimBlock>>>(n, N_THREAD, inter, cscColPtr);
    CHECK_CUDA_ERROR
}

/// =============================================================
/// ========================================== Apply `prefix_sum`
/// =============================================================

void ScanTransposer::prefix_sum(int n, int *cscColPtr) {
    for(int i = 0; i < n; i++) {
        cscColPtr[i+1] += cscColPtr[i];
    }
}

/// =============================================================
/// ============================================ Reorder elements
/// =============================================================

__global__ 
void reorder_elements_kernel(
    int n, int nnz, int len, 
    int *inter, int *intra, 
    int *csrRowIdx, int *csrColIdx, float *csrVal,
    int *cscColPtr, int *cscRowIdx, float *cscVal
) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global_id
    int start = tid * len;

    for(int i = 0; i < len && start + i < nnz; i++) {
        int cid = csrColIdx[start + i];
        int loc = cscColPtr[cid] + inter[tid*n + cid] + intra[start + i];
        cscRowIdx[loc] = csrRowIdx[start + i];
        cscVal[loc] = csrVal[start + i];
    }
}

void ScanTransposer::reorder_elements_caller(
    int n, int nnz, int *inter, int *intra, 
    int *csrRowIdx, int *csrColIdx, float *csrVal,
    int *cscColPtr, int *cscRowIdx, float *cscVal
) {

    dim3 DimGrid(DIV_THEN_CEIL(N_THREAD, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    const int LEN = DIV_THEN_CEIL(nnz, N_THREAD);

    reorder_elements_kernel<<<DimGrid, DimBlock>>>(n, nnz, LEN, inter, intra, 
        csrRowIdx, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);
    CHECK_CUDA_ERROR
}

/// =============================================================
/// =============================================================
/// =============================================================

int ScanTransposer::csr2csc_gpumemory(int m, int n, int nnz, int *csrRowPtr, int *csrColIdx, float *csrVal, int *cscColPtr, int *cscRowIdx, float *cscVal)
{
    int *intra, *inter, *csrRowIdx;

    // resource allocation
    cudaMalloc(&csrRowIdx, nnz*sizeof(int)); // no need to set `csrRowIdx` to zeros
    cudaMalloc(&intra,     nnz*sizeof(int));
    cudaMalloc(&inter,     ((N_THREAD + 1) * n)*sizeof(int));
    SAFE_CALL(cudaMemset(intra, 0, (nnz)*sizeof(int)))
    SAFE_CALL(cudaMemset(inter, 0, ((N_THREAD + 1) * n)*sizeof(int)))

    // 1. fill `csrRowIdx`
    //std::cout << "1. cscrowidx_caller" << std::endl;
    csrrowidx_caller(m, csrRowPtr, csrRowIdx);
    if(SCANTRANS_DEBUG_ENABLE) {
        int* csrRowIdx_host = new int[nnz];
        SAFE_CALL(cudaMemcpy(csrRowIdx_host, csrRowIdx, nnz*sizeof(int), cudaMemcpyDeviceToHost));
        print_array("csrRowIdx", csrRowIdx_host, nnz);
        delete csrRowIdx_host;
    }

    // 2. fill `inter`, `intra`
    //std::cout << "2. inter_intra_caller" << std::endl;
    inter_intra_caller(n, nnz, inter, intra, csrColIdx);
    if(SCANTRANS_DEBUG_ENABLE) {
        int* intra_host = new int[nnz];
        int* inter_host = new int[nnz];
        SAFE_CALL(cudaMemcpy(intra_host, intra, nnz*sizeof(int),              cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(inter_host, inter, ((N_THREAD+1)*n)*sizeof(int), cudaMemcpyDeviceToHost));
        print_array("intra", intra_host, nnz);
        std::cout << "inter: " << std::endl;
        for(int i = 0; i < N_THREAD+1; i++) {
            std::cout << "Row " << (i-1) << ": ";
            for(int j = 0; j < n; j++) {
                std::cout << inter_host[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
        delete intra_host;
        delete inter_host;
    }

    // 3. apply vertical scan
    //std::cout << "3. vertical_scan_caller" << std::endl;
    vertical_scan_caller(n, inter, cscColPtr);
    if(SCANTRANS_DEBUG_ENABLE) {
        int* inter_host = new int[nnz];
        SAFE_CALL(cudaMemcpy(inter_host, inter, ((N_THREAD+1)*n)*sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "inter: " << std::endl;
        for(int i = 0; i < N_THREAD+1; i++) {
            std::cout << "Row " << (i-1) << ": ";
            for(int j = 0; j < n; j++) {
                std::cout << inter_host[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
        delete inter_host;
    }

    // 4. apply prefix sum
    {
        int *cscColPtr_host = new int[n+1];
        SAFE_CALL(cudaMemcpy(cscColPtr_host, cscColPtr, (n+1)*sizeof(int), cudaMemcpyDeviceToHost));
        cscColPtr_host[0] = 0;
        prefix_sum(n, cscColPtr_host);
        SAFE_CALL(cudaMemcpy(cscColPtr, cscColPtr_host, (n+1)*sizeof(int), cudaMemcpyHostToDevice));
        if(SCANTRANS_DEBUG_ENABLE) {
            print_array("cscColPtr", cscColPtr_host, n+1);
        }
        delete cscColPtr_host;
    }

    // 5. reorder elements
    reorder_elements_caller(n, nnz, inter, intra, csrRowIdx, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);

    // deallocate resources
    cudaFree(csrRowIdx);
    cudaFree(intra);
    cudaFree(inter); 

    return COMPUTATION_OK;
}