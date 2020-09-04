#include <iostream>
#include <iomanip>
#include "helper_cuda.h"
#include "tester/Tester.hh"

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

    Tester tester;
    tester.add_test_instance(5000, 5000, 1000000, 5);
    bool has_error = tester.run();
    std::cout << "Has error: " << has_error << std::endl;
    tester.print();

    return 0;
}