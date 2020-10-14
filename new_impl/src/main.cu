#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include "matrix.hh"
#include "procedures.hh"
#include "Timer.cuh"
using namespace timer;

const int ITERATION_NUMBER = 5;

int main(int argc, char **argv) {

    matrix::SparseMatrix * sm;
    std::string filename;

    // inizializzazione della matrice
    if(argc > 1) {

        // leggo file mtx esterno
        filename = std::string(argv[1]);
        std::ifstream file(filename);
        sm = new matrix::SparseMatrix(file);
        file.close();
    } else {

        // matrice generata casualmente
        filename = "random";
        sm = new matrix::SparseMatrix(100'000, 100'000, 10'000'000);
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

            //std::cout << "\nProcessing " << i << std::flush;
            // modalità nel quale sto trasponendo
            matrix::TranspositionMethod tm = (matrix::TranspositionMethod) i;
            
            // traspongo (timer)
            timers[i].start();
            matrix::SparseMatrix * smt = sm->transpose(tm);
            timers[i].stop();

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    }

    // salvo in output il tempo impiegato
    for(int i = matrix::SERIAL; i <= matrix::CUSPARSE2; i++) {
        std::cout << timers[i].average() << "; ";
    }

    std::cout << "\n";

    return 0;
}