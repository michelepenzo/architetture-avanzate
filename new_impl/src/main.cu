#include <iostream>
#include <iomanip>
#include "matrix.hh"
#include "transposer.hh"

int main(int argc, char **argv) {

    matrix::SparseMatrix *s = new matrix::SparseMatrix(5, 5, 10);
    s->print();

    matrix::SparseMatrix *st = transposer::transpose(s, transposer::SERIAL);
    st->print();

    matrix::SparseMatrix *st2 = transposer::transpose(s, transposer::MERGE);
    st2->print();

    // deallocation
    if(s != NULL)  delete s;
    if(st != NULL) delete st;
    if(st2 != NULL) delete st2;
    return 0;
}