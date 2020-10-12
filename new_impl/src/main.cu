#include <iostream>
#include <iomanip>
#include <fstream>
#include "matrix.hh"
#include "procedures.hh"
#include "Timer.cuh"
using namespace timer;

const int ITERATION_NUMBER = 1000;

int main(int argc, char **argv) {

    matrix::SparseMatrix * sm;
    std::string filename;

    for(int i = 0; i < argc; i++) {
        std::cout << i << " " << argv[i] << "\n";
    }

    // inizializzazione della matrice
    if(argc > 1) {

        // leggo file mtx esterno
        filename = std::string(argv[1]);
        std::cout << "Reading filename " << filename << "\n";
        std::ifstream file(filename);
        sm = new matrix::SparseMatrix(file);
    } else {

        // matrice generata casualmente
        filename = "random";
        std::cout << "Reading random\n";
        sm = new matrix::SparseMatrix(100'000, 100'000, 1'000'000);
    }

    // stampa dei dati della matrice
    std::cout << filename << "; ";
    std::cout << sm->m << "; ";
    std::cout << sm->n << "; ";
    std::cout << sm->nnz << "; ";

    // esecuzione della trasposta
    Timer<HOST> timers[5];
    for(int j = 0; j < ITERATION_NUMBER; j++) {

        for(int i = matrix::SERIAL; i <= matrix::CUSPARSE2; i++) {

            // modalità nel quale sto trasponendo
            matrix::TranspositionMethod tm = (matrix::TranspositionMethod) i;
            
            // traspongo (timer)
            timers[i].start();
            matrix::SparseMatrix * smt = sm->transpose(tm);
            timers[i].stop();
        }

    }

    // salvo in output il tempo impiegato
    for(int i = matrix::SERIAL; i <= matrix::CUSPARSE2; i++) {
        std::cout << timers[i].average() << "; ";
    }

    return 0;
}