#pragma once

#include <cstdio>
#include <tuple>
#include <vector>
#include "../Timer.cuh"
using namespace timer;

#include "../matrices/SparseMatrix.hh"
#include "../transposers/SerialTransposer.hh"
#include "../transposers/CusparseTransposer.hh"
#include "../transposers/ScanTransposer.hh"

/// Contains matrices with error
struct ErrorMatrices {

    SparseMatrix *reference, *serial, *cusparse, *scantrans;

    ErrorMatrices() : reference(0), serial(0), cusparse(0), scantrans(0) {}

    ErrorMatrices(SparseMatrix* ref, SparseMatrix* ser, SparseMatrix* cus, SparseMatrix* sct) {
        reference = ref;
        serial = ser;
        cusparse = cus;
        scantrans = sct;
    }

    ~ErrorMatrices() {
        if(reference != 0) { delete reference; }
        if(serial != 0)    { delete serial; }
        if(cusparse != 0)  { delete cusparse; }
        if(scantrans != 0) { delete scantrans; }
    }
};

/// Contains the necessary data to generate a single test instance
/// and its results. 
struct TestInstance {

    int m;

    int n;

    int nnz;

    int repetitions;

    float mean_serial_timing;

    float mean_cusparse_timing;

    float mean_scantrans_timing;

    std::vector<ErrorMatrices> errors;

    TestInstance(int m, int n, int nnz, int rep) : errors() {
        this->m = m;
        this->n = n;
        this->nnz = nnz;
        this->repetitions = rep;
    }
};

/// Run the tests
class Tester {
private:

    std::vector<TestInstance> test_instances;

    Timer<HOST> timer_serial;

    Timer<DEVICE> timer_cusparse;

    Timer<DEVICE> timer_scantrans;

public:

    Tester(): timer_serial(), timer_cusparse(), timer_scantrans(), test_instances() { }

    void add_test(int m, int n, int nnz, int rep) {
        test_instances.push_back(TestInstance(m, n, nnz, rep));
    }

    /// Run each of the test you've added previously
    /// @return: false if we have without error
    bool run() {

        bool any_error = false;

        // run each single test instance
        for(TestInstance& test: test_instances) {

            // foreach repetition run each transposer
            for(int i = 0; i < test.repetitions; i++) {
                // see progress on screen
                std::cout << "." << std::flush;

                // create this random matrix
                SparseMatrix* sm = new SparseMatrix(test.m, test.n, test.nnz, RANDOM_INITIALIZATION);

                // create transposer objects
                SerialTransposer serial_transposer(sm);
                CusparseTransposer cusparse_transposer(sm);
                ScanTransposer scantrans_transposer(sm);

                // run SERIAL transposition
                timer_serial.start();
                SparseMatrix* serial_sm = serial_transposer.transpose();
                timer_serial.stop();

                // run CUSPARSE transposition
                timer_cusparse.start();
                SparseMatrix* cusparse_sm = cusparse_transposer.transpose();
                timer_cusparse.stop();

                // run SCAN TRANS transposition
                timer_scantrans.start();
                SparseMatrix* scantrans_sm = scantrans_transposer.transpose();
                timer_scantrans.stop();

                // check if there is any error (compare to reference impl 'Serial')
                bool error = false;
                if(cusparse_sm == NULL || !serial_sm->equals(cusparse_sm)) {
                    error = true;
                    test.errors.push_back(ErrorMatrices(sm, serial_sm, cusparse_sm, scantrans_sm));
                }
                if(scantrans_sm == NULL || !serial_sm->equals(scantrans_sm)) {
                    error = true;
                    test.errors.push_back(ErrorMatrices(sm, serial_sm, cusparse_sm, scantrans_sm));
                }
                any_error = any_error || error;

                // deallocate resources only without any error
                if(!error) {
                    delete sm;
                    delete serial_sm;
                    delete cusparse_sm;
                    delete scantrans_sm;
                }
            }

            // at the end of each repetition, save time
            test.mean_serial_timing = timer_serial.average(); 
            test.mean_cusparse_timing = timer_cusparse.average();
            test.mean_scantrans_timing = timer_scantrans.average();

            // reset timers
            timer_serial.reset();
            timer_cusparse.reset();
            timer_scantrans.reset();
        }

        // newline after the points
        std::cout << std::endl;

        return any_error;
    }

    /// Print table with average execution time and speedup
    void print() {

        printf("╔═════════════════════════════╤═══════════════════════════╤═══════════════════════════╗\n");
        printf("║ %-27s │ %-25s │ %-25s ║\n", "TEST SPECS", "SERIAL", "CUSPARSE");
        printf("╟────────┬────────┬───────────┼───────────────┬───────────┼───────────────┬───────────╢\n");
        printf("║ %-6s │ %-6s │ %-9s │ %-13s │ %-9s │ %-13s │ %-9s ║\n", "M", "N", "NNZ", "MEAN", "SPEEDUP", "MEAN", "SPEEDUP");
        printf("╟────────┼────────┼───────────┼───────────────┼───────────┼───────────────┼───────────╢\n");
        for(TestInstance const& test: test_instances) {
            // calculate speedups
            float cusparse_speedup = test.mean_serial_timing / test.mean_cusparse_timing;
            // print row
            printf("║ %6i │ %6i │ %9i │ %13.5f │ %8.2fx │ %13.5f │ %8.2fx ║\n", test.m, test.n, test.nnz, 
                test.mean_serial_timing, 1.0f, test.mean_cusparse_timing, cusparse_speedup);
        }
        printf("╚════════╧════════╧═══════════╧═══════════════╧═══════════╧═══════════════╧═══════════╝\n");       
    }

};
