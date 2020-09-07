#include <iostream>
#include <iomanip>
#include "helper_cuda.h"
#include "tester/Tester.hh"
#include "transposers/ScanTransposer.hh"

#define REPETITION_NUMBER 1

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

//  Tester tester;
//  tester.add_test(   10,    10,       20, REPETITION_NUMBER);
//  tester.add_test(  100,   100,     1000, REPETITION_NUMBER);
//  tester.add_test( 1000,  1000,    10000, REPETITION_NUMBER);
//  tester.add_test(10000, 10000,  1000000, REPETITION_NUMBER);
//  tester.add_test(10000, 10000, 10000000, REPETITION_NUMBER);
//  tester.run();
//  tester.print();

    SparseMatrix* sm = new SparseMatrix(5, 5, 10, RANDOM_INITIALIZATION);
    sm->print();

    ScanTransposer sctr(sm);
    SparseMatrix* sm_transposed = sctr.transpose();
    if( sm_transposed != NULL ) {
        sm_transposed->print();
    } 

    delete sm;
    delete sm_transposed;

    return 0;
}