#pragma once

#include "tester_fn.hh"
#include "utilities.hh"

class tester_segsort : public tester_fn<int> {

    int BLOCK_SIZE;

    size_t initialize(int instance_number, int ** input) override {

        size_t len = (size_t) instance_number;

        int * _input = new int[len];
        *input = _input;

        for(int i = 0; i < len; i++) {
            _input[i] = utils::random::generate(20);
        }

        return len;
    }

    void call_reference(int * output, int * input, size_t len) {
        procedures::reference::segsort(input, output, len);
    }

    void call_cuda(int * output, int * input, size_t len) {
        procedures::cuda::segsort(input, output, len);
    }

    bool is_ok(int * ref, int * cud, size_t len) override {
        return utils::equals<int>(ref, cud, len);
    }

public:
    tester_segsort() : tester_fn("segsort") { }
};