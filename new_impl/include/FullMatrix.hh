#pragma once

#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include "AbstractMatrix.hh"
#include "SparseMatrix.hh"

class FullMatrix : public AbstractMatrix {

private:

    void fill_with_rand_numbers() {
        // 1. init risorse random
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::uniform_int_distribution<int> distribution(1, 100);

        // 2. creo la matrice "classica"
        // 2.1 riempio i primi `nnz` elementi
        for (int i = 0; i < nnz; i++)
            matrix[i] = distribution(generator);

        // 2.2 mischio gli elementi nell'array
        std::shuffle(matrix, matrix+(n*m), generator);
    }

public:

    float* matrix;
    

    FullMatrix(const int m, const int n, const int nnz) : AbstractMatrix(m, n, nnz) {

        this->matrix = new float[n*m]();

        fill_with_rand_numbers();
    }

    FullMatrix(const SparseMatrix& sm) : AbstractMatrix(sm.m, sm.n, sm.nnz) {

        matrix = new float[n*m]();

        // 1. genero indici di riga, in questo modo la coppia (row_indices, indices) mi porta ad avere la notazione COO
        int* csrRowIdx = new int[nnz];
        // 2. riempio l'array con gli indici di riga
        for(int i = 0; i < m; i++) {
            int row_start = sm.csrRowPtr[i], row_end = sm.csrRowPtr[i+1];
            for(int j = row_start; j < row_end; j++) {
                csrRowIdx[j] = i;
            }
        }

        // sistemo gli elementi nella matrice
        for(int i = 0; i < nnz; i++) {
            // estrai la colonna
            int col = sm.csrColIdx[i];
            // estrai la riga
            int row = csrRowIdx[i];
            // salvo il valore nella matrice di output
            matrix[row*n + col] = sm.csrVal[i];
        }

        delete[] csrRowIdx;

    }

    ~FullMatrix() {
        delete[] this->matrix;
    }

    void print() {

        std::cout << std::endl << "matrix: " << std::endl;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                std::cout << std::setw(3) << matrix[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    SparseMatrix* to_sparse() {

        SparseMatrix* sm = new SparseMatrix(m, n, nnz, ALL_ZEROS_INITIALIZATION);

        // 1. riempio i campi
        int i = 0;
        for(int row = 0; row < m; row++) {

            // devo aggiungere il primo elemento di ogni riga
            bool primoElementoRigaTrovato = false;

            for(int col = 0; col < n; col++) {
                
                // prendo il dato
                float cell = matrix[row * n + col];

                // se il dato non è nullo lo aggiungo
                if(cell != 0) {
                    sm->csrVal[i] = cell;
                    sm->csrColIdx[i] = col;

                    // se è il primo elemento, lo salvo nei puntatori agli indici
                    if( !primoElementoRigaTrovato ) {
                        sm->csrRowPtr[row] = i;
                        primoElementoRigaTrovato = true;
                    }

                    // incremento puntatore alla prossima cella di `data`
                    i++;
                }
            }

            // se non ho trovato il primo elemento della riga, allora metto 
            // il valore segnaposto -1 che viene poi sistemato successivamente
            if( !primoElementoRigaTrovato ) {
                sm->csrRowPtr[row] = -1;
            }
        }

        // 2. sistemo l'ultimo elemento di `indptr` 
        sm->csrRowPtr[m] = nnz;
        
        // 3. sistemo l'anomalia dei -1 nell'array indptr
        for(int i = m; i > 0; i--) {
            if(sm->csrRowPtr[i-1] == -1) {
                sm->csrRowPtr[i-1] = sm->csrRowPtr[i];
            }
        }

        return sm;
    }

};