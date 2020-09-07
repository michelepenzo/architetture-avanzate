#include <iostream>
#include <iomanip>
#include "helper_cuda.h"
#include "libfort/fort.hpp"
#include "tester/Tester.hh"
#include "transposers/ScanTransposer.hh"
#include "transposers/CusparseTransposer.hh"

#define REPETITION_NUMBER 100

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

    CusparseTransposer cu;
    ScanTransposer sc;
    ScanTransposer sc2(256, 1024);
    ScanTransposer sc3(256, 1536);
    Tester tester;
    tester.add_test(   10,    10,       20, REPETITION_NUMBER);
    tester.add_test(  100,   100,     1000, REPETITION_NUMBER);
    tester.add_test( 1000,  1000,    10000, REPETITION_NUMBER);
    tester.add_test(10000, 10000,  1000000, REPETITION_NUMBER);
    tester.add_test(10000, 10000, 10000000, REPETITION_NUMBER);
    tester.add_processor(&cu, "CUSPARSE");
    tester.add_processor(&sc, "SCAN256-256");
    tester.add_processor(&sc2, "SCAN256-1k");
    tester.add_processor(&sc3, "SCAN256-1.5k");
    tester.run();
    tester.print();

    // ScanTransposer sc(256, 5);
    // Tester tester;
    // tester.add_test(5, 4, 10, REPETITION_NUMBER);
    // tester.add_processor(&sc, "SCANTRANS");
    // tester.run(true);
    // tester.print();
    
    return 0;
}