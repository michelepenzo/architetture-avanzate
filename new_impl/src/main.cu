#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include "SparseMatrix.hh"
#include "FullMatrix.hh"

int main() {

    SparseMatrix* sm = new SparseMatrix(4, 5, 4);
    sm->print();

    FullMatrix* fm = new FullMatrix(*sm);
    fm->print();

    SparseMatrix* sm2 = fm->to_sparse();
    sm2->print();

    delete sm, fm, sm2;
    return 0;
}