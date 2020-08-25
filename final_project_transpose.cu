#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include "Timer.cuh"
#include "CheckError.cuh"
using namespace timer;

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


int main() {

    int m = 4, n = 6, nnz = 15;
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
    csr2csc_serial(m, n, nnz, csrRowPtr, csrColIdx, csrVal, cscColPtr, cscRowIdx, cscVal);
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