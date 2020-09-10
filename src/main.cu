#include <iostream>
#include <iomanip>
#include "helper_cuda.h"
#include "libfort/fort.hpp"
#include "cuda_utils/prefix_scan.hh"
#include "tester/Tester.hh"
#include "transposers/ScanTransposer.hh"
#include "transposers/CusparseTransposer.hh"

#define REPETITION_NUMBER 1

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

    const int N_ELEM = 17;
    int *input_array = new int[N_ELEM]();
    int *output_array = new int[N_ELEM]();
    std::cout << "Input : ";
    for(int i = 0; i < N_ELEM; i++) {
        input_array[i] = i;
        std::cout << std::setw(2) << input_array[i] << " ";
    }


    scan(output_array, input_array, N_ELEM, true);


    std::cout << "\nOutput: ";
    for(int i = 0; i < N_ELEM; i++) {
        output_array[i] = i;
        std::cout << std::setw(2) << output_array[i] << " ";
    }

    std::cout << "\n\n";





    //CusparseTransposer cu;
    //ScanTransposer sc;
    //ScanTransposer sc2(256, 1024);
    //ScanTransposer sc3(256, 1536);
    //Tester tester;
    ////tester.add_test(   10,    10,       20, REPETITION_NUMBER);
    ////tester.add_test(  100,   100,     1000, REPETITION_NUMBER);
    ////tester.add_test( 1000,  1000,    10000, REPETITION_NUMBER);
    //tester.add_test(17, 17,  20, REPETITION_NUMBER);
    //tester.add_processor(&cu, "CUSPARSE");
    //tester.add_processor(&sc, "SCAN256-256");
    ////tester.add_processor(&sc2, "SCAN256-1k");
    ////tester.add_processor(&sc3, "SCAN256-1.5k");
    //tester.run(false);
    //tester.print();
//
    //// ScanTransposer sc(256, 256);
    //// Tester tester;
    //// tester.add_test(5000, 4000, 1000, REPETITION_NUMBER);
    //// tester.add_processor(&sc, "SCANTRANS");
    //// tester.run(true);
    //// tester.print();
    
    return 0;
}