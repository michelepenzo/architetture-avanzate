#pragma once
#ifndef MATRIX_HH_
#define MATRIX_HH_

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <set>
#include <fstream>
#include "utilities.hh"
#include "procedures.hh"
#include "transposers.hh"

namespace matrix {

    enum MatrixInitialization {
        ALL_ZEROS_INITIALIZATION    = 0,
        RANDOM_INITIALIZATION       = 1
    };

    enum TranspositionMethod {
        SERIAL      = 0,
        SCANTRANS   = 1, 
        MERGETRANS  = 2,
        CUSPARSE1   = 3,
        CUSPARSE2   = 4
    };

    // rappresentazione della matrice in formato CSR
    class SparseMatrix {

    public:
        // dimensioni della matrice
        int m, n;
        
        // numero di elementi non nulli
        int nnz;
        
        // puntatore agli elementi di inizio riga
        int * csrRowPtr;
        
        // array dell'indice di colonna degli elementi
        int * csrColIdx;
        
        // array dei valori degli elementi
        float * csrVal;
        
        // costruttore, se mi=RANDOM_INITIALIZATION allora genera matrice causale
        SparseMatrix(const int m, const int n, const int nnz, const MatrixInitialization mi = RANDOM_INITIALIZATION);
        
        // costruttore da file MTX
        SparseMatrix(std::ifstream& mtx_file);

        // distruttore
        ~SparseMatrix();
        
        // true se le matrici sono uguali
        bool equals(SparseMatrix* sm);

        // ritorna la matrice trasposta attraverso il metodo deciso da tm
        SparseMatrix* transpose(TranspositionMethod tm);
    };

    // rappresentazione della matrice in formato esteso
    class FullMatrix {

        // inizializzazione degli elementi
        void fill_with_rand_numbers();

    public:
        // dimensioni della matrice
        int m, n;
        
        // numero di elementi non nulli
        int nnz;

        // array degli elementi m * n
        float* matrix;

        // costruttore, se mi=RANDOM_INITIALIZATION allora genera matrice causale
        FullMatrix(const int m, const int n, const int nnz, const MatrixInitialization mi = RANDOM_INITIALIZATION);

        // costruttore, inizializza partendo dalla matrice sparsa
        FullMatrix(const SparseMatrix* sm);

        // distruttore
        ~FullMatrix();

        // true se le matrici sono uguali
        bool equals(FullMatrix* fm);

        // traspone la matrice
        FullMatrix* transpose();

        // restituisce una copia della matrice in formato CSR
        SparseMatrix* to_sparse();
    };
}

#endif