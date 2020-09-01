#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include "Timer.cuh"
#include "CheckError.cuh"
#define COMPUTATION_ERROR -1
#define COMPUTATION_OK 0
using namespace timer;


//#include <cublas_v2.h>
//#include <cusparse.h>
//#include <curand.h>
//#include <cuda_runtime_api.h>
//
///* Need openacc for the stream definitions */
//#include "openacc.h"
//
//int csr2csc_cusparse_internal(
//        int m, int n, int nnz, 
//        int* csrRowPtr, int* csrColIdx, float* csrVal, 
//        int* cscColPtr, int* cscRowIdx, float* cscVal) {
//
//    cusparseHandle_t handle;
//    cusparseStatus_t status;
//    size_t buffer_size;
//    
//    // 1. allocate resources
//    status = cusparseCreate(&handle);
//    if(status != CUSPARSE_STATUS_SUCCESS) {
//        std::cerr << "csr2csc_cusparse - Error while calling cusparseCreate: " << cusparseGetErrorName(status) << std::endl;
//        return COMPUTATION_ERROR;
//    }
//
//    status = cusparseSetStream(handle, (cudaStream_t) acc_get_cuda_stream(acc_async_sync));
//    if(status != CUSPARSE_STATUS_SUCCESS) {
//        std::cerr << "csr2csc_cusparse - Error while calling cusparseSetStream: " << cusparseGetErrorName(status) << std::endl;
//        return COMPUTATION_ERROR;
//    }
//
//    // 2. ask cusparse how much space it needs to operate
//    status = cusparseCsr2cscEx2_bufferSize(
//        handle,                     // link to cusparse engine
//        m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscColPtr, cscRowIdx, 
//        CUDA_R_32F,                 // [valType] data type of csrVal, cscVal arrays is 32-bit real (non-complex) single precision floating-point
//        CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
//        CUSPARSE_INDEX_BASE_ZERO,   // [idxBase]
//        CUSPARSE_CSR2CSC_ALG1,      // which algorithm is used? CUSPARSE_CSR2CSC_ALG1 or CUSPARSE_CSR2CSC_ALG2
//        &buffer_size);              // fill buffer_size variable
//
//    if(status != CUSPARSE_STATUS_SUCCESS) {
//        std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2_bufferSize: " << cusparseGetErrorName(status) << std::endl;
//        cusparseDestroy(handle);
//        return COMPUTATION_ERROR;
//    } else if(buffer_size <= 0) {
//        std::cerr << "csr2csc_cusparse - warning: buffer_size is not positive" << std::endl;
//    }
//
//    // 3. callocate buffer space
//    void* buffer = NULL;
//    SAFE_CALL( cudaMalloc(&buffer, buffer_size) );
//    std::cout << "Needed " << buffer_size << " bytes to esecute Csr2csc" << std::endl; 
//
//    // 4. call transpose
//    status = cusparseCsr2cscEx2(
//        handle, 
//        m, n, nnz, csrVal, csrRowPtr, csrColIdx, cscVal, cscColPtr, cscRowIdx, 
//        CUDA_R_32F,                 // [valType] data type of csrVal, cscVal arrays is 32-bit real (non-complex) single precision floating-point
//        CUSPARSE_ACTION_NUMERIC,    // [copyValues] the operation is performed on data and indices.
//        CUSPARSE_INDEX_BASE_ZERO,   // [idxBase]
//        CUSPARSE_CSR2CSC_ALG1,      // which algorithm is used? CUSPARSE_CSR2CSC_ALG1 or CUSPARSE_CSR2CSC_ALG2
//        buffer);                    // cuda buffer
//
//    if(status != CUSPARSE_STATUS_SUCCESS) {
//        std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2: " << cusparseGetErrorName(status) << std::endl;
//        std::cerr << "csr2csc_cusparse - Error while calling cusparseCsr2cscEx2: " << cusparseGetErrorString(status) << std::endl;
//        SAFE_CALL( cudaFree( buffer ) );
//        cusparseDestroy(handle);
//        return COMPUTATION_ERROR;
//    }
//
//    SAFE_CALL( cudaFree( buffer ) );
//    status = cusparseDestroy(handle);
//    if(status != CUSPARSE_STATUS_SUCCESS) {
//        std::cerr << "csr2csc_cusparse - Error while calling cusparseDestroy: " << cusparseGetErrorName(status) << std::endl;
//        return COMPUTATION_ERROR;
//    }
//
//    return COMPUTATION_OK;
//}


// int csr2csc_cusparse(
//         int m, int n, int nnz, 
//         int* csrRowPtr, int* csrColIdx, float* csrVal, 
//         int* cscColPtr, int* cscRowIdx, float* cscVal) {
// 
//     int* csrRowPtr_dev;
//     int* csrColIdx_dev;
//     float* csrVal_dev;
//     int* cscColPtr_dev;
//     int* cscRowIdx_dev;
//     float* cscVal_dev;
// 
//     cudaMalloc(&csrRowPtr_dev,(m+1)*sizeof(int));
//     cudaMalloc(&csrColIdx_dev,(nnz)*sizeof(int));
//     cudaMalloc(&csrVal_dev   ,(nnz)*sizeof(float));
//     cudaMalloc(&cscColPtr_dev,(n+1)*sizeof(int));
//     cudaMalloc(&cscRowIdx_dev,(nnz)*sizeof(int));
//     cudaMalloc(&cscVal_dev   ,(nnz)*sizeof(float));
// 
//     int ret = csr2csc_cusparse_internal(m, n, nnz, csrRowPtr_dev, csrColIdx_dev, csrVal_dev, cscColPtr_dev, cscRowIdx_dev, cscVal_dev);
// 
//     cudaFree(csrRowPtr_dev);
//     cudaFree(csrColIdx_dev);
//     cudaFree(csrVal_dev);
//     cudaFree(cscColPtr_dev);
//     cudaFree(cscRowIdx_dev);
//     cudaFree(cscVal_dev);
// 
//     return ret;
// }

void create_matrix_full(int m, int n, int nnz, float* full_matrix) {

    // 1. init risorse random
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    // 2. creo la matrice "classica"
    // 2.1 riempio i primi `nnz` elementi
    for (int i = 0; i < nnz; i++)
        full_matrix[i] = distribution(generator);

    // 2.2 mischio gli elementi nell'array
    std::shuffle(full_matrix, full_matrix+(n*m), generator);
}

void print_matrix_full(int m, int n, float* full_matrix) {

    std::cout << std::endl << "matrix: " << std::endl;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << std::setw(3) << full_matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

void create_matrix_csr(int m, int n, int nnz, float* full_matrix, int* csrRowPtr, int* csrColIdx, float* csrVal) {

    // 1. riempio i campi
    int i = 0;
    for(int row = 0; row < m; row++) {

        // devo aggiungere il primo elemento di ogni riga
        bool primoElementoRigaTrovato = false;

        for(int col = 0; col < n; col++) {
            
            // prendo il dato
            float cell = full_matrix[row * n + col];

            // se il dato non è nullo lo aggiungo
            if(cell != 0) {
                csrVal[i] = cell;
                csrColIdx[i] = col;

                // se è il primo elemento, lo salvo nei puntatori agli indici
                if( !primoElementoRigaTrovato ) {
                    csrRowPtr[row] = i;
                    primoElementoRigaTrovato = true;
                }

                // incremento puntatore alla prossima cella di `data`
                i++;
            }
        }

        // se non ho trovato il primo elemento della riga, allora metto 
        // il valore segnaposto -1 che viene poi sistemato successivamente
        if( !primoElementoRigaTrovato ) {
            csrRowPtr[row] = -1;
        }
    }

    // 2. sistemo l'ultimo elemento di `indptr` 
    csrRowPtr[m] = nnz;
    
    // 3. sistemo l'anomalia dei -1 nell'array indptr
    for(int i = m; i > 0; i--) {
        if(csrRowPtr[i-1] == -1) {
            csrRowPtr[i-1] = csrRowPtr[i];
        }
    }
}

void print_matrix_csr(int m, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal) {

    std::cout << std::endl << "csrRowPtr: ";
    for(int i = 0; i < m + 1; i++) {
        std::cout << csrRowPtr[i] << " ";
    }

    std::cout << std::endl << "csrColIdx: ";
    for(int i = 0; i < nnz; i++) {
        std::cout << csrColIdx[i] << " ";
    }

    std::cout << std::endl << "csrVal: ";
    for(int i = 0; i < nnz; i++) {
        std::cout << csrVal[i] << " ";
    }

    std::cout << std::endl;
}

void create_matrix_from_csr(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, float* full_matrix) {

    // genero indici di riga, in questo modo la coppia (row_indices, indices) mi porta ad avere la notazione COO
    int* csrRowIdx = new int[nnz];
    // riempio l'array con gli indici di riga
    for(int i = 0; i < m; i++) {
        int row_start = csrRowPtr[i], row_end = csrRowPtr[i+1];
        for(int j = row_start; j < row_end; j++) {
            csrRowIdx[j] = i;
        }
    }

    // sistemo gli elementi nella matrice
    for(int i = 0; i < nnz; i++) {
        // estrai la colonna
        int col = csrColIdx[i];
        // estrai la riga
        int row = csrRowIdx[i];
        // salvo il valore nella matrice di output
        full_matrix[row*n + col] = csrVal[i];
    }

    delete[] csrRowIdx;
}

void csr2csc_serial(
        int m, int n, int nnz, 
        int* csrRowPtr, int* csrColIdx, float* csrVal, 
        int* cscColPtr, int* cscRowIdx, float* cscVal) {

    int* curr = new int[n]();
    // 1. costruisco `cscColPtr` come istogramma delle frequenze degli elementi per ogni colonna
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            cscColPtr[csrColIdx[j]+1]++;
        }
    }
    // 2. applico prefix_sum per costruire corretto `cscColPtr` (ogni cella tiene conto dei precedenti)
    for(int i = 1; i < n+1; i++) {
        cscColPtr[i] += cscColPtr[i-1];
    }
    // 3. sistemo indici di riga e valori
    for(int i = 0; i < m; i++) {
        for(int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++) {
            int col = csrColIdx[j];
            int loc = cscColPtr[col] + curr[col];
            curr[col]++;
            cscRowIdx[loc] = i;
            cscVal[loc] = csrVal[j];
        }
    }

    delete[] curr;
}



void print_gpu_infos() {

    int driverVersion = -1;
    SAFE_CALL( cudaDriverGetVersion(&driverVersion));
    std::cout << "CUDA Driver version: " << driverVersion << std::endl;

    int runtimeVersion = -1;
    SAFE_CALL( cudaRuntimeGetVersion(&runtimeVersion));
    std::cout << "CUDA Runtime version: " << runtimeVersion << std::endl;

    int dev_count = -1;
    cudaGetDeviceCount(&dev_count);

    for (int i = 0; i < dev_count; i++) {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties( &dev_prop, i);
        
        std::cout << "maxThreadsPerBlock: " << dev_prop.maxThreadsPerBlock << std::endl;
        std::cout << "sharedMemPerBlock: " << dev_prop.sharedMemPerBlock << std::endl;
        std::cout << "Compute capability: " << dev_prop.major << std::endl;
    }
}


__global__
void scan_trans_kernel(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, 
                       int* cscColPtr, int* cscRowIdx, float* cscVal, int* inter, int* intra){
    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // intra e inter
    if(global_id < nnz){
        int index = (global_id + 1) * n + csrColIdx[global_id];
        intra[global_id]=inter[index];     
        inter[index] = inter[index]+1;
    }    
}



void scan_trans(int m, int n, int nnz, int* csrRowPtr, int* csrColIdx, float* csrVal, int* cscColPtr, int* cscRowIdx, float* cscVal){

    int* intra;
    int* inter;
    //int* csrRowIdx_dev;
    int* csrRowPtr_dev;
    int* csrColIdx_dev;
    float* csrVal_dev;
    int* cscColPtr_dev;
    int* cscRowIdx_dev;
    float* cscVal_dev;


    const int N = nnz;              // n_thread
    const int BLOCK_SIZE_X = 256;        
    //const int BLOCK_SIZE_y = 16;

    cudaMalloc(&intra,    (nnz)*sizeof(int));
    cudaMalloc(&inter,    ((N+1)*n)*sizeof(int));
    //cudaMalloc(&csrRowIdx_dev,(nnz)*sizeof(int));
    cudaMalloc(&csrColIdx_dev,(nnz)*sizeof(int));
    cudaMalloc(&csrVal_dev,   (nnz)*sizeof(float));
    cudaMalloc(&csrRowPtr_dev,(m+1)*sizeof(int));
    cudaMalloc(&cscColPtr_dev,(n+1)*sizeof(int));
    cudaMalloc(&cscRowIdx_dev,(nnz)*sizeof(int));
    cudaMalloc(&cscVal_dev,   (nnz)*sizeof(float));

    SAFE_CALL(cudaMemcpy(csrRowPtr_dev, csrRowPtr, (m+1)*sizeof(int), cudaMemcpyHostToDevice)); 
    SAFE_CALL(cudaMemcpy(csrColIdx_dev, csrColIdx, (nnz)*sizeof(int), cudaMemcpyHostToDevice)); 
    SAFE_CALL(cudaMemcpy(csrVal_dev,    csrVal,    (nnz)*sizeof(int), cudaMemcpyHostToDevice)); 
    //SAFE_CALL(cudaMemcpy(csrRowIdx, csrRowIdx_dev, (nnz)*sizeof(int), cudaMemcpyHostToDevice)); 
    
    // DEVICE INIT
    dim3 DimGrid(N/BLOCK_SIZE_X, 1, 1);
    if (N%BLOCK_SIZE_X) DimGrid.x++;
    //if (N%BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, 1, 1);
    
    // -------------------------------------------------------------------------
    // DEVICE EXECUTION

    // KERNEL
    scan_trans_kernel<<<DimGrid,DimBlock>>> (m, n, nnz, csrRowPtr_dev, csrColIdx_dev, csrVal_dev, cscColPtr_dev, cscRowIdx_dev, cscVal_dev, inter, intra);

    CHECK_CUDA_ERROR

    int* intra_host = new int[nnz];
    int* inter_host = new int[(N+1)*n];

    SAFE_CALL(cudaMemcpy(cscRowIdx,  cscRowIdx_dev, (m+1)*sizeof(int),     cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscColPtr,  cscColPtr_dev, (nnz)*sizeof(int),     cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(cscVal,     cscVal_dev,    (nnz)*sizeof(int),     cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(intra_host, intra,         (nnz)*sizeof(int),     cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(inter_host, inter,         ((N+1)*n)*sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Intra: " << std::endl;
    for(int i = 0; i < nnz; i++) {
        std::cout << intra_host[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Inter: ";
    for(int i = 0; i < (N+1); i++) {
        std::cout << std::endl << "Row " << i-1 << ": ";
        for(int j = 0; j < n; j++) {
            std::cout << inter_host[i * n + j] << std::endl;
        }
    }
    std::cout << std::endl;



    cudaFree(intra);
    cudaFree(inter);
    cudaFree(csrRowPtr_dev);
    cudaFree(csrColIdx_dev);
    cudaFree(csrVal_dev);
    cudaFree(cscColPtr_dev);
    cudaFree(cscRowIdx_dev);
    cudaFree(cscVal_dev);

    delete[] intra_host;
    delete[] inter_host;

    return;
}



int main() {

    print_gpu_infos();
    std::cout << std::endl;

    int m = 4, n = 4, nnz = 10;
    // matrice "full"
    float* full_matrix = new float[n*m]();
    // matrice csr normale
    int* csrRowPtr = new int[m+1]();
    int* csrColIdx = new int[nnz]();
    float* csrVal = new float[nnz]();
    // matrice csr trasposta (csc)
    int* cscColPtr = new int[n+1]();
    int* cscRowIdx = new int[nnz]();
    float* cscVal = new float[nnz]();
    // matrice "full" per controllo 
    float* full_matrix_check = new float[n*m]();


    // 1. creo la matrice "full"
    create_matrix_full(m, n, nnz, full_matrix);
    print_matrix_full(m, n, full_matrix);

    // 2. converto in formato CSR
    create_matrix_csr(m, n, nnz, full_matrix, csrRowPtr, csrColIdx, csrVal);
    print_matrix_csr(m, nnz, csrRowPtr, csrColIdx, csrVal);

    // 3. traspongo (seriale)
    scan_trans(m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);
    print_matrix_csr(n, nnz, cscColPtr, cscRowIdx, cscVal); // (!) ora ho invertito `n` ed `m`

    // 4. riconverto in matrice `full`
    create_matrix_from_csr(n, m, nnz, cscColPtr, cscRowIdx, cscVal, full_matrix_check); // (!) ora ho invertito `n` ed `m`
    print_matrix_full(n, m, full_matrix_check);

    // 5. controllo correttezza della trasposta
    int error = 0;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if(full_matrix[i*n+j] != full_matrix_check[j*m+i]) {
                std::cout << "Errore all'indice: " << i << std::endl;
                error++;
            }
        }
    }
    if(error == 0) {
        std::cout << "Non ci sono errori" << std::endl;
    }

    delete[] full_matrix;
    delete[] full_matrix_check;
    delete[] csrRowPtr;
    delete[] csrColIdx;
    delete[] csrVal;
    delete[] cscColPtr;
    delete[] cscRowIdx;
    delete[] cscVal;

    return 0;
}