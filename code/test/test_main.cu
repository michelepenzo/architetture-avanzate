#include <iostream>
#include <iomanip>
#include <set>
#include <utility>
#include "matrix.hh"
#include "procedures.hh"
#include "transposers.hh"

#include "tester_fn.hh"
#include "tester_merge.hh"
#include "tester_matrix.hh"
#include "tester_pti.hh"
#include "tester_itp.hh"
#include "tester_segsort.hh"

#include "Timer.cuh"
using namespace timer;

int main(int argc, char **argv) {

    // TIMING
    // Timer<HOST> timer;
    // timer.start();
    // timer.stop();
    // timer.average() - .duration()
    // timer.reset()

    tester_segsort tsegs;
    tester_pti tpti;
    tester_itp titp;
    tester_matrix tx;
    tester_merge tm;
    tester_merge3 tm3;
    
    bool ok = true;
    ok &= tpti.test_many_instances();
    //ok &= tx.test_many_instances();
    //ok &= tm3.test_many_instances();
    //ok &= tm.test_many_instances();

    std::cout << "ESITO: ";
    std::cout << (ok ? "OK" : "NO") << std::endl;

    return 0;
}
