#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include "matrix.hh"
#include "procedures.hh"
#include "Timer.cuh"
using namespace timer;

const int ITERATION_NUMBER = 5;

int main(int argc, char **argv) {

    matrix::SparseMatrix * sm;
    std::string filename;

    if(argc == 1) {
        // matrice generata casualmente con dimensione fissa
        filename = "random";
        sm = new matrix::SparseMatrix(100'000, 100'000, 10'000'000);

    } else if(argc == 2) {
        // matrice importata da file MTX
        filename = std::string(argv[1]);
        std::ifstream file(filename);
        if(!file.good()) {
            throw std::invalid_argument("Cannot open given file " + filename);
        }
        sm = new matrix::SparseMatrix(file);
        file.close();

    } else if(argc == 4) {
        filename = "random_from_specs";
        int m = std::stoi(argv[1]), n = std::stoi(argv[2]), nnz = std::stoi(argv[3]);
        sm = new matrix::SparseMatrix(m, n, nnz);

    } else {
        throw std::invalid_argument("Invalid number of arguments: " + std::to_string(argc-1) + ", must be 0 or 1 or 3");
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