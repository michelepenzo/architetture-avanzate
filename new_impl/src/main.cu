#include <iostream>
#include <iomanip>
#include "helper_cuda.h"
#include "tester/Tester.hh"

#define REPETITION_NUMBER 10

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

    Tester tester;
    tester.add_test(10, 10, 20, REPETITION_NUMBER);
    tester.add_test(100, 100, 1000, REPETITION_NUMBER);
    tester.add_test(1000, 1000, 10000, REPETITION_NUMBER);
    tester.add_test(10000, 10000, 1000000, REPETITION_NUMBER);
    tester.add_test(10000, 10000, 10000000, REPETITION_NUMBER);
    tester.run();
    tester.print();

    return 0;
}