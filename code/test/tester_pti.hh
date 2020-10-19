#pragma once
#include "tester.hh"
#include "procedures.hh"

class tester_pti : public tester {

    bool test_instance(int instance_number) override {
        int m = instance_number;

        // generate input
        int * input = utils::random::generate_array<int>(0, 3, m+1);
        input[m] = 0;
        DPRINT_MSG("...")
        utils::prefix_sum(input, m+1);
        DPRINT_ARR(input, m+1)

        // get nnz
        int nnz = input[m];

        // run reference implementation
        int * reference_output = new int[nnz](); // init to zeros
        procedures::reference::pointers_to_indexes(input, m, reference_output, nnz);
        
        // run parallel implementation
        int * parallel_output      = new int[nnz];
        int * parallel_cuda_input  = utils::cuda::allocate_send<int>(input, m+1);
        int * parallel_cuda_output = utils::cuda::allocate_zero<int>(nnz);
        procedures::cuda::pointers_to_indexes(parallel_cuda_input, m, parallel_cuda_output, nnz);
        utils::cuda::recv(parallel_output, parallel_cuda_output, nnz);

        // check correctness
        bool ok = utils::equals<int>(reference_output, parallel_output, nnz);
        DPRINT_ARR(reference_output, nnz)
        DPRINT_ARR(parallel_output, nnz)

        utils::cuda::deallocate(parallel_cuda_input);
        utils::cuda::deallocate(parallel_cuda_output);
        delete input, reference_output, parallel_output;

        return ok;
    }

public:
    tester_pti() : tester("ptr_to_idx") { }
};