#pragma once
#include "tester.hh"
#include "procedures.hh"

class tester_itp : public tester {

    bool test_instance(int instance_number) override {

        int NNZ = instance_number, N = utils::random::generate(instance_number*2)+1;

        int * colIdx = utils::random::generate_array<int>(0, N-1, NNZ);
        int * inter;
        int * intra = new int[NNZ]();
        int * colPtr = new int[N+1]();

        int * colIdx_cuda = utils::cuda::allocate_send<int>(colIdx, NNZ);
        int * inter_cuda;
        int * intra_cuda = utils::cuda::allocate_zero<int>(NNZ);
        int * colPtr_cuda = utils::cuda::allocate_zero<int>(N+1);

        int * inter_cuda_out  = new int[(HISTOGRAM_BLOCKS+1) * N];
        int * intra_cuda_out  = new int[NNZ];
        int * colPtr_cuda_out = new int[N+1];

        DPRINT_ARR(colIdx, NNZ);
        procedures::reference::indexes_to_pointers(colIdx, NNZ, &inter, intra, colPtr, N);
        for(int i = 0; i < HISTOGRAM_BLOCKS+1; i++) {
            DPRINT_ARR(inter+i*N, N);
        }
        DPRINT_ARR(intra, NNZ);
        DPRINT_ARR(colPtr, N+1);

        procedures::cuda::indexes_to_pointers(colIdx_cuda, NNZ, &inter_cuda, intra_cuda, colPtr_cuda, N);
        utils::cuda::recv<int>(inter_cuda_out, inter_cuda, (HISTOGRAM_BLOCKS+1) * N);
        utils::cuda::recv<int>(intra_cuda_out, intra_cuda, NNZ);
        utils::cuda::recv<int>(colPtr_cuda_out, colPtr_cuda, N+1);
        for(int i = 0; i < HISTOGRAM_BLOCKS; i++) {
            DPRINT_ARR(inter_cuda_out+i*N, N);
        }
        DPRINT_ARR(intra_cuda_out, NNZ);
        DPRINT_ARR(colPtr_cuda_out, N+1);

        utils::cuda::deallocate(colIdx_cuda);
        utils::cuda::deallocate(inter_cuda);
        utils::cuda::deallocate(intra_cuda);
        utils::cuda::deallocate(colPtr_cuda);

        bool ok = utils::equals(inter, inter_cuda_out, (HISTOGRAM_BLOCKS+1)*N)
            || utils::equals(intra, intra_cuda_out, NNZ)
            || utils::equals(colPtr, colPtr_cuda_out, N+1);

        delete[] colIdx, inter, intra, colPtr;
        delete[] inter_cuda_out, intra_cuda_out, colPtr_cuda_out;

        return ok;

    }

public:

    tester_itp() : tester("idx_to_ptr") { }
};