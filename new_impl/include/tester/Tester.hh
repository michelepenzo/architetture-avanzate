#pragma once

#include <tuple>
#include <vector>
#include "../Timer.cuh"
#include "../matrices/SparseMatrix.hh"
#include "../transposers/SerialTransposer.hh"
#include "../transposers/CusparseTransposer.hh"
using namespace timer;


/// Contain statistics about the running time of each *Transposer
struct Timing {
    float mean, std, min, max;
    Timing() {} 
    Timing(float me, float s, float mi, float ma) {
        mean = me;
        std = s;
        min = mi;
        max = ma;
    }
};

/// Contains matrices with error
struct ErrorMatrices {
    SparseMatrix *reference, *serial, *cusparse;
    ErrorMatrices() : reference(0), serial(0), cusparse(0) {}
    ErrorMatrices(SparseMatrix* ref, SparseMatrix* ser, SparseMatrix* cus) {
        reference = ref;
        serial = ser;
        cusparse = cus;
    }
};

/// Contains the necessary data to generate a single test instance
/// and its results. 
struct TestInstance {

    int m;

    int n;

    int nnz;

    int repetitions;

    Timing serial_timing;

    Timing cusparse_timing;

    std::vector<ErrorMatrices> errors;

    TestInstance(int m, int n, int nnz, int rep) : 
        serial_timing(),
        cusparse_timing(),
        errors()
    {
        this->m = m;
        this->n = n;
        this->nnz = nnz;
        this->repetitions = rep;
    }

    bool has_error() {
        return errors.size() == 0;
    }
};

/// Run the tests
class Tester {
private:

    std::vector<TestInstance> test_instances;

    Timer<HOST> timer_serial;

    Timer<DEVICE> timer_cusparse;

public:

    Tester(): timer_serial(), timer_cusparse(), test_instances() { 

    }

    void add_test_instance(int m, int n, int nnz, int rep) {
        test_instances.push_back(TestInstance(m, n, nnz, rep));
    }

    /// 
    /// @return: false if we have without error
    bool run() {

        bool any_error = false;

        // run each single test instance
        for(TestInstance& test: test_instances) {

            // foreach repetition run each transposer
            for(int i = 0; i < test.repetitions; i++) {

                // create this random matrix
                SparseMatrix* sm = new SparseMatrix(test.m, test.n, test.nnz, RANDOM_INITIALIZATION);

                // create transposer objects
                SerialTransposer serial_transposer(sm);
                CusparseTransposer cusparse_transposer(sm);

                // run SERIAL transposition
                timer_serial.start();
                SparseMatrix* serial_sm = serial_transposer.transpose();
                timer_serial.stop();

                // run CUSPARSE transposition
                timer_cusparse.start();
                SparseMatrix* cusparse_sm = cusparse_transposer.transpose();
                timer_cusparse.stop();

                // check if there is any error (compare to reference impl 'Serial')
                bool error = false;
                any_error = any_error || error;
                if(! serial_sm->equals(cusparse_sm)) {
                    error = true;
                    test.errors.push_back(ErrorMatrices(sm, serial_sm, cusparse_sm));
                }

                // deallocate resources only without any error
                if(!error) {
                    delete sm;
                    delete serial_sm;
                    delete cusparse_sm;
                }
            }

            // at the end of each repetition, save time
            test.serial_timing = Timing(timer_serial.average(), timer_serial.std_deviation(), timer_serial.min(), timer_serial.max());
            test.cusparse_timing = Timing(timer_cusparse.average(), timer_cusparse.std_deviation(), timer_cusparse.min(), timer_cusparse.max());

            // reset timers
            timer_serial.reset();
            timer_cusparse.reset();
        }

        return any_error;
    }

    void print() {

        std::cout << "*** Tester ***" << std::endl;

        for(TestInstance const& test: test_instances) {

            std::cout << "Run instance with m=" << test.m << ", n=" << test.n << ", nnz=" << test.nnz << std::endl;
            std::cout << "Serial:   mean=" << test.serial_timing.mean << std::endl;
            std::cout << "Cusparse: mean=" << test.cusparse_timing.mean << std::endl;
            std::cout << std::endl;
        }
    }

};
