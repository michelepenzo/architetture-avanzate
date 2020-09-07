#include <iostream>
#include <iomanip>
#include "helper_cuda.h"
#include "libfort/fort.hpp"
#include "tester/Tester.hh"
#include "transposers/ScanTransposer.hh"
#include "transposers/CusparseTransposer.hh"

#define REPETITION_NUMBER 1

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

    // CusparseTransposer cu;
    // ScanTransposer sc;
    // Tester tester;
    // tester.add_test(   10,    10,       20, REPETITION_NUMBER);
    // tester.add_test(  100,   100,     1000, REPETITION_NUMBER);
    // tester.add_test( 1000,  1000,    10000, REPETITION_NUMBER);
    // tester.add_test(10000, 10000,  1000000, REPETITION_NUMBER);
    // tester.add_processor(&cu, "CUSPARSE");
    // tester.add_processor(&sc, "SCANTRANS");
    // tester.run();
    // tester.print();

    ScanTransposer sc;
    Tester tester;
    tester.add_test(5, 4, 10, REPETITION_NUMBER);
    tester.add_processor(&sc, "SCANTRANS");
    tester.run(true);
    tester.print();
    
    return 0;
}