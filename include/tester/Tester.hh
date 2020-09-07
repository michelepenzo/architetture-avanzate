#pragma once

#include <cstdio>
#include <tuple>
#include <string>
#include <vector>
#include "../Timer.cuh"
using namespace timer;

#include "../matrices/SparseMatrix.hh"
#include "../transposers/AbstractTransposer.hh"
#include "../transposers/SerialTransposer.hh"

/// Run the tests
class Tester {
private:

    struct TestSpecification {
        int m, n, nnz, repetitions;
    };

    struct Processor {
        AbstractTransposer *at;
        std::string name;
        Timer<HOST> timer;
        bool any_error;
    };

    std::vector<TestSpecification> test_specifications;

    std::vector<Processor> processors;

    float** mean_timing;

    void allocate_mean_timing() {
        size_t ntest = test_specifications.size();
        size_t nproc = processors.size();
        mean_timing = new float*[ntest];
        for(int i = 0; i < ntest; i++) {
            mean_timing[i] = new float[nproc];
        }
    }

    void deallocate_mean_timing() {
        if(mean_timing != NULL) {
            for(int i = 0; i < test_specifications.size(); i++) {
                if(mean_timing[i] != NULL) { 
                    delete[] mean_timing[i]; 
                    mean_timing[i] = NULL; 
                }
            }
            delete[] mean_timing;
            mean_timing = NULL;
        }
    }

public:

    Tester(): test_specifications(), processors(), mean_timing(NULL) { 
        add_processor(new SerialTransposer(), "SERIAL");
    }

    ~Tester() { 
        deallocate_mean_timing();
    }

    void add_test(int m, int n, int nnz, int rep) {
        TestSpecification ts = { .m = m, .n = n, .nnz = nnz, .repetitions = rep };
        test_specifications.push_back(ts);
    }

    void add_processor(AbstractTransposer *at, std::string name) {
        Timer<HOST> timer;
        Processor pr = {.at = at, .name = name, .timer = timer, .any_error = false};
        processors.push_back(pr);
    }

    bool run(bool debug=false) {

        // keep `any_error` variable
        bool any_error = false;

        // initialize `mean_timing` structure
        deallocate_mean_timing();
        allocate_mean_timing();        

        // start execution
        size_t test_index = 0;
        for(const TestSpecification& test : test_specifications) {
            
            // PERFORMANCE EVALUATION
            for(int _i = 0; _i < test.repetitions; _i++) {

                // create this random matrix
                SparseMatrix* sm = new SparseMatrix(test.m, test.n, test.nnz, RANDOM_INITIALIZATION);

                // run reference implementation
                processors[0].timer.start();
                SparseMatrix* transposed_sm = processors[0].at->transpose(sm);
                processors[0].timer.stop();

                // print infos 
                if(debug) {
                    std::cout << "ORIGINAL" << std::endl;
                    sm->print();
                    std::cout << "REFERENCE" << std::endl;
                    transposed_sm->print();
                } else {
                    // see progress on screen
                    std::cout << "." << std::flush;
                }

                // run any other implementation
                for(int j = 1; j < processors.size(); j++) {

                    // print infos
                    if(debug) {
                        std::cout << processors[j].name << std::endl;
                    }

                    // run with timer
                    processors[j].timer.start();
                    SparseMatrix* other_sm = processors[j].at->transpose(sm);
                    processors[j].timer.stop();

                    // check error
                    if(other_sm == NULL) {
                        processors[j].any_error = true;
                        any_error = true;
                        if(debug) { std::cout << "computation error" << std::endl; }
                    } else if(!transposed_sm->equals(other_sm)) {
                        processors[j].any_error = true;
                        any_error = true;
                        if(debug) { std::cout << "computation error" << std::endl; }
                        if(debug) { other_sm->print(); }
                        delete other_sm;
                    } else {
                        if(debug) { other_sm->print(); }
                        delete other_sm;
                    }
                }

                // deallocate resource
                delete transposed_sm;
                delete sm;

                // print info
                if(debug) {
                    std::cout << std::endl;
                }
            }

            // SAVE TIMINGS
            for(int i = 0; i < processors.size(); i++) {
                mean_timing[test_index][i] = processors[i].timer.average();
                processors[i].timer.reset();
            }
            
            test_index++;
        }

        return any_error;
    }


    /// Print table with average execution time and speedup
    void print() {

        fort::char_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // print (first) header
        table << fort::header << "SPECIFICATION" << "" << "";
        table[0][0].set_cell_span(3);
        int i = 0;
        for(const Processor& proc : processors) {
            table << proc.name << "";
            table[0][3+i*2].set_cell_span(2);
            i++;
        }
        table << fort::endr;

        // print (second) header
        table << fort::header << "M" << "N" << "NNZ";
        for(const Processor& proc : processors) {
            table << "MEAN TIME" << "SPEEDUP";
        }
        table << fort::endr;

        // print values
        for(int i = 0; i < test_specifications.size(); i++) {

            // print specs
            TestSpecification& test = test_specifications[i];
            table << test.m << test.n << test.nnz;

            // print timing
            for(int j = 0; j < processors.size(); j++) {
                Processor& proc = processors[j];
                if(proc.any_error) {
                    table << "error" << "error";
                } else {
                    table 
                        << std::fixed << std::setprecision(3) << mean_timing[i][j]
                        << std::fixed << std::setprecision(3) << (mean_timing[i][0] / mean_timing[i][j]);
                }
            }

            table << fort::endr;
        }

        // setting alignment
        for(int i = 0; i < 3 + 2*processors.size(); i++) {
            table.column(i).set_cell_text_align(fort::text_align::right);
        }

        std::cout << std::endl << table.to_string() << std::endl;
    }

};
