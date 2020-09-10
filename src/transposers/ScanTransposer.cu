#include "transposers/ScanTransposer.hh"
#include "cuda_utils/prefix_scan.hh"

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
        cscColPtr[i] = inter[nthread*n+i]; // cscColPtr[i+1] = inter[nthread][i];
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
    for(int i = n-1; i >= 0; i--) {
        cscColPtr[i+1] = cscColPtr[i];
    }
    cscColPtr[0] = 0;
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

int ScanTransposer::csr2csc_gpumemory(
    int m, int n, int nnz, 
    int *csrRowPtr, int *csrColIdx, float *csrVal, 
    int *cscColPtr, int *cscRowIdx, float *cscVal,
    int* csrRowPtr_host, int* csrColIdx_host, float* csrVal_host, 
    int* cscColPtr_host, int* cscRowIdx_host, float* cscVal_host)
{
    int *intra, *inter, *csrRowIdx;
    int *intra_host, *inter_host, *csrRowIdx_host;
    
#if SCANTRANS_DEBUG_ENABLE==1
    // allocate var for comparison 
    int *inter_temp = new int[(N_THREAD+1)*n]();
    int *intra_temp = new int[nnz]();
    int *cscColPtr_temp = new int[n+1]();
#endif

    // resource allocation
    csrRowIdx_host = new int[nnz];
    intra_host = new int[nnz];
    inter_host = new int[(N_THREAD + 1) * n];
    cudaMalloc(&csrRowIdx, nnz*sizeof(int)); // no need to set `csrRowIdx` to zeros
    cudaMalloc(&intra,     nnz*sizeof(int));
    cudaMalloc(&inter,     ((N_THREAD + 1) * n)*sizeof(int));
    SAFE_CALL(cudaMemset(intra, 0, (nnz)*sizeof(int)))
    SAFE_CALL(cudaMemset(inter, 0, ((N_THREAD + 1) * n)*sizeof(int)))

    // 1. fill `csrRowIdx`
    csrrowidx_caller(m, csrRowPtr, csrRowIdx);

    // debug check
    if(SCANTRANS_DEBUG_ENABLE) {
        std::cout << "Step 1: csrrowidx_caller" << std::endl;
        // retrieve value from GPU
        SAFE_CALL(cudaMemcpy(csrRowIdx_host, csrRowIdx, nnz*sizeof(int), cudaMemcpyDeviceToHost));
        // compare with serial 
        for(int i = 0; i < m; i++) {
            for(int j = csrRowPtr_host[i]; j < csrRowPtr_host[i+1]; j++) {
                if(csrRowIdx_host[j] != i) {
                    std::cout << "Error: i=" << i << ", j=" << j 
                              << ", csrRowPtr_host[i]=" 
                              << csrRowPtr_host[i]
                              << ", csrRowPtr_host[i+1]="
                              << csrRowPtr_host[i+1]
                              << ", csrRowIdx_host[j]="
                              << csrRowIdx_host[j] << std::endl;
                    return COMPUTATION_ERROR;
                }
            }
        }
    }

    // 2. fill `inter`, `intra`
    inter_intra_caller(n, nnz, inter, intra, csrColIdx);

    // debug check
    if(SCANTRANS_DEBUG_ENABLE) {

        std::cout << "Step 2: inter_intra_caller" << std::endl;

        // retrieve values
        SAFE_CALL(cudaMemcpy(intra_host, intra, nnz*sizeof(int),              cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(inter_host, inter, ((N_THREAD+1)*n)*sizeof(int), cudaMemcpyDeviceToHost));

        for(int tid = 0; tid < N_THREAD; tid++) {

            int len = DIV_THEN_CEIL(nnz, N_THREAD);
            int start = tid * len;
            
            for(int i = 0; i < len && start + i < nnz; i++) {
                int index = (tid + 1) * n + csrColIdx_host[start + i];
                intra_temp[start + i] = inter_temp[index];
                inter_temp[index]++;
            }
        }
        
        for(int i = 0; i < nnz; i++) {
            if(intra_host[i] != intra_temp[i]) {
                std::cout << "Error on intra_host index " << i << " value=" << intra_host[i] << " expected=" << intra_temp[i] << std::endl;
                return COMPUTATION_ERROR;
            }
        }

        for(int i = 0; i < (N_THREAD+1)*n; i++) {
            if(inter_host[i] != inter_temp[i]) {
                std::cout << "Error on inter_host index " << i << " value=" << inter_host[i] << " expected=" << inter_temp[i] << std::endl;
                return COMPUTATION_ERROR;
            }
        }
        
       
    }

    // 3. apply vertical scan
    vertical_scan_caller(n, inter, cscColPtr);

    if(SCANTRANS_DEBUG_ENABLE) {
        std::cout << "Step 3: vertical_scan_caller" << std::endl;

        SAFE_CALL(cudaMemcpy(inter_host, inter, ((N_THREAD+1)*n)*sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(cscColPtr_host, cscColPtr, (n+1)*sizeof(int), cudaMemcpyDeviceToHost));
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < N_THREAD; j++) {
                inter_temp[(j+1)*n+i] += inter_temp[j*n+i]; // inter[j+1][i] += inter[j][i];
            
                if(inter_host[(j+1)*n+i] != inter_temp[(j+1)*n+i]) {
                    std::cout << "Error on inter_host index " << ((j+1)*n+i) << " value=" << inter_host[(j+1)*n+i] << " expected=" << inter_temp[(j+1)*n+i] << std::endl;
                    return COMPUTATION_ERROR;
                }
            
            }
            cscColPtr_temp[i] = inter_temp[N_THREAD*n+i]; // cscColPtr[i+1] = inter[nthread][i];
        }
    }

    // 4. apply prefix sum
    std::cout << "Step 4: scan_on_cuda" << std::endl;
    scan_on_cuda(cscColPtr, cscColPtr, n, true);

    if(SCANTRANS_DEBUG_ENABLE) {
        SAFE_CALL(cudaMemcpy(cscColPtr_host, cscColPtr, (n+1)*sizeof(int), cudaMemcpyDeviceToHost));
        
        prefix_sum(n, cscColPtr_temp);

        std::cout << "### scan_on_cuda: cscColPtr_ : ";
        for(int i = 0; i < n+1; i++) {
            std::cout << std::setw(2) << cscColPtr_temp[i] << " ";
        }
        std::cout << "\n";
        
        for(int i = 0; i < n+1; i++) {
            if(cscColPtr_host[i] != cscColPtr_temp[i]) {
                std::cout << "Error on cscColPtr_host index " << i << " value=" << cscColPtr_host[i] << " expected=" << cscColPtr_temp[i] << std::endl;
                return COMPUTATION_ERROR;
            }
        }
    }

    // 5. reorder elements
    reorder_elements_caller(n, nnz, inter, intra, csrRowIdx, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);

    // deallocate resources
    cudaFree(csrRowIdx);
    cudaFree(intra);
    cudaFree(inter); 
    delete intra_host;
    delete inter_host;
    delete csrRowIdx_host;

#if SCANTRANS_DEBUG_ENABLE==1
    delete intra_temp;
    delete inter_temp;
    delete cscColPtr_temp;
#endif

    return COMPUTATION_OK;
}