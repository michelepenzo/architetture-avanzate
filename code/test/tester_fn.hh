#include "tester.hh"
#include "procedures.hh"
#include "merge_step.hh"

NUMERIC_TEMPLATE(T)
class tester_fn : public tester {

    virtual size_t initialize(int instance_number, T ** input) = 0;

    virtual void call_reference(T * output, T * input, size_t len) = 0;

    virtual void call_cuda(T * output, T * input, size_t len) = 0;

    virtual bool is_ok(T * ref, T * cud, size_t len) = 0;

    bool test_instance(int instance_number) override {

        // inizializzazione dei dati
        T * input;
        size_t len = initialize(instance_number, &input);
        T * ref_output = new int[len]();
        T * cuda_output = new int[len]();
        DPRINT_ARR(input, len)

        // call reference implementation
        call_reference(ref_output, input, len);
        DPRINT_ARR(ref_output, len)

        // call cuda implementation
        T * dev_cuda_input  = utils::cuda::allocate_send<T>(input, len);
        T * dev_cuda_output = utils::cuda::allocate<T>(len);
        call_cuda(dev_cuda_output, dev_cuda_input, len);
        utils::cuda::recv(cuda_output, dev_cuda_output, len);
        DPRINT_ARR(cuda_output, len)

        bool ok = is_ok(ref_output, cuda_output, len);

        // deallocazione
        delete[] input, ref_output, cuda_output;
        utils::cuda::deallocate(dev_cuda_input);
        utils::cuda::deallocate(dev_cuda_output);
        return ok;
    }

public:
    tester_fn(std::string name) : tester(name) { }
};






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
        procedures::reference::segmerge_step(input, output, len, BLOCK_SIZE);
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
        procedures::reference::segmerge3_step(
            input, output, 
            len, BLOCK_SIZE,
            val1_input, ref_val1_output,
            val2_input, ref_val2_output
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
