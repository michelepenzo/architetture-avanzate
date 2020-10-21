#pragma once 
#include "tester_fn.hh"

class tester_merge : public tester_fn<int> {

    int BLOCK_SIZE;

    size_t initialize(int instance_number, int ** input) override {

        size_t len = (size_t) instance_number;

        BLOCK_SIZE = utils::next_two_pow(DIV_THEN_CEIL(len, 10));

        int * _input = new int[len];
        *input = _input;

        for(int blk = 0; blk < DIV_THEN_CEIL(len, BLOCK_SIZE); blk++) {
            const int K = len - blk*BLOCK_SIZE;
            
            for(int i = 0; i < BLOCK_SIZE; i++) {
                if(blk*BLOCK_SIZE + i >= len) { return len; }
                _input[blk*BLOCK_SIZE + i] = K + i; 
            }
        }

        return len;
    }

    void call_reference(int * output, int * input, size_t len) {
        procedures::reference::merge_step<int>(input, output, len, BLOCK_SIZE);
    }

    void call_cuda(int * output, int * input, size_t len) {
        procedures::cuda::merge_step<int>(input, output, len, BLOCK_SIZE);
    }

    bool is_ok(int * ref, int * cud, size_t len) override {
        return utils::equals<int>(ref, cud, len);
    }

public:
    tester_merge() : tester_fn("merge") { }
};





class tester_merge3 : public tester_fn<int> {

    int* val1_input, *ref_val1_output, *cuda_val1_output;
    float* val2_input, *ref_val2_output, *cuda_val2_output;

    int BLOCK_SIZE;

    size_t initialize(int instance_number, int ** input) override {

        size_t len = (size_t) instance_number;
        BLOCK_SIZE = utils::next_two_pow(DIV_THEN_CEIL(len, 10));

        int * _input = new int[len];
        *input = _input;

             val1_input  = new int[len];
         ref_val1_output = new int[len];
        cuda_val1_output = new int[len];
             val2_input  = new float[len];
         ref_val2_output = new float[len];
        cuda_val2_output = new float[len];

        for(int blk = 0; blk < DIV_THEN_CEIL(len, BLOCK_SIZE); blk++) {
            const int K = len - blk*BLOCK_SIZE;
            
            for(int i = 0; i < BLOCK_SIZE; i++) {
                if(blk*BLOCK_SIZE + i >= len) { return len; }
                _input[blk*BLOCK_SIZE + i] = K + i; 
                val1_input[blk*BLOCK_SIZE + i] = K + i + 1; 
                val2_input[blk*BLOCK_SIZE + i] = K + i + 2; 
            }
        }

        return len;
    }

    void call_reference(int * output, int * input, size_t len) {
        procedures::reference::merge3_step(
            input, output, 
            val1_input, ref_val1_output,
            val2_input, ref_val2_output,
            len, BLOCK_SIZE
        );
    }

    void call_cuda(int * output, int * input, size_t len) {
        int   * dev_val1 = utils::cuda::allocate_send<int>(val1_input, len);
        float * dev_val2 = utils::cuda::allocate_send<float>(val2_input, len);
        int   * dev_val1_out = utils::cuda::allocate<int>(len);
        float * dev_val2_out = utils::cuda::allocate<float>(len);
        procedures::cuda::merge3_step<int>(
            input, output, 
            dev_val1, dev_val1_out,
            dev_val2, dev_val2_out,
            len, BLOCK_SIZE
        );
        CUDA_SAFE_CALL(cudaMemcpy(cuda_val1_output, dev_val1_out, len*sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(cuda_val2_output, dev_val2_out, len*sizeof(float), cudaMemcpyDeviceToHost));
        utils::cuda::deallocate(dev_val1);
        utils::cuda::deallocate(dev_val2);
        utils::cuda::deallocate(dev_val1_out);
        utils::cuda::deallocate(dev_val2_out);
    }

    bool is_ok(int * ref, int * cud, size_t len) override {
        bool ok = utils::equals<int>(ref, cud, len) 
            && utils::equals<int>(ref_val1_output, cuda_val1_output, len)
            && utils::equals<float>(ref_val2_output, cuda_val2_output, len);
        delete[] val1_input, ref_val1_output, cuda_val1_output;
        delete[] val2_input, ref_val2_output, cuda_val2_output;
        return ok;
    }

public:
    tester_merge3() : tester_fn("merge3") { }
};