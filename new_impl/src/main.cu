#include <iostream>
#include <iomanip>
#include "matrix.hh"
#include "transposer.hh"

int main(int argc, char **argv) {

    bool ok = false;
    /* matrix::SparseMatrix *s = new matrix::SparseMatrix(5, 5, 10);
    s->print();

    matrix::SparseMatrix *st = transposer::transpose(s, transposer::SERIAL);
    st->print();

    matrix::SparseMatrix *st2 = transposer::transpose(s, transposer::MERGE);
    st2->print();

    // deallocation
    if(s != NULL)  delete s;
    if(st != NULL) delete st;
    if(st2 != NULL) delete st2;

    ok = transposer::component_test::sort();
    std::cout << "OK: " << ok << "\n";
    */

 // ok = transposer::component_test::segmerge();
 // std::cout << "segmerge     OK: " << ok << "\n";

 // ok = transposer::component_test::segmerge3();
 // std::cout << "segmerge3    OK: " << ok << "\n";

//  ok = transposer::component_test::segmerge_sm();
//  std::cout << "segmerge_sm  OK: " << ok << "\n";
    
    ok = transposer::component_test::segmerge3_sm();
    std::cout << "segmerge3_sm OK: " << ok << "\n";

//  ok = transposer::component_test::sort();
//  std::cout << "sort OK: " << ok << "\n";

    return 0;
}