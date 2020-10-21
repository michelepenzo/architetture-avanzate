#pragma once
#include "tester.hh"
#include "procedures.hh"

class tester_matrix : public tester {
public:
    tester_matrix() : tester("matrix") { }

    bool test_instance(int instance_number) {
        if(instance_number > TESTER_SMALL_INSTANCES_MAX) {
            return true;
        }

        bool all_ok = true;

        int NNZ = instance_number;
        int N = instance_number, M = instance_number;

        matrix::FullMatrix * fm   = new matrix::FullMatrix(M, N, NNZ);
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        matrix::FullMatrix * fmt  = fm->transpose();
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
        matrix::SparseMatrix * sm = fm->to_sparse();
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        for(int i = matrix::TranspositionMethod::MERGETRANS; i <= matrix::TranspositionMethod::MERGETRANS; ++i) {
            
            matrix::TranspositionMethod tm = static_cast<matrix::TranspositionMethod>(i);
            //std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            matrix::SparseMatrix * smt = sm->transpose(tm);
            //std::this_thread::sleep_for(std::chrono::milliseconds(100));

            matrix::FullMatrix * fmtt = new matrix::FullMatrix(smt);
            //std::this_thread::sleep_for(std::chrono::milliseconds(100));

            all_ok &= fmt->equals(fmtt);
            delete smt, fmtt;
        }
        
        delete fm, fmt, sm;

        return all_ok;
    }
};
