#pragma once
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
