#include <iostream>
#include <iomanip>
#include "matrices/SparseMatrix.hh"
#include "matrices/FullMatrix.hh"
#include "transposers/SerialTransposer.hh"
#include "transposers/CusparseTransposer.hh"
#include "helper_cuda.h"

int main(int argc, char **argv) {

    findCudaDevice(argc, (const char **) argv);

    SparseMatrix* sm = new SparseMatrix(4, 5, 4);
    sm->print();

    FullMatrix* fm = new FullMatrix(*sm);
    fm->print();

    CusparseTransposer transposer(sm);

    SparseMatrix* sm_tr = transposer.transpose();
    sm_tr->print();

    FullMatrix* fm_tr = new FullMatrix(*sm_tr);
    fm_tr->print();
    
    delete sm, fm, sm_tr, fm_tr;
    return 0;
}