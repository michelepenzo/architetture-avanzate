#include <iostream>
#include <iomanip>
#include <set>
#include <utility>
#include "matrix.hh"
#include "procedures.hh"
#include "transposers.hh"

#define TESTER_ALL_INSTANCES_MIN 1
#define TESTER_ALL_INSTANCES_MAX 20'000
#define TESTER_BIG_INSTANCES 100'000'000

class tester {

    virtual bool test_instance(int instance_number) = 0;

public:

    bool test_many_instances() {
        
        bool all_ok = true;
        for(int m = TESTER_ALL_INSTANCES_MIN; m < TESTER_ALL_INSTANCES_MAX; m++) {
            std::cout << "Testing with m=" << std::setw(10) << m << ": " << std::flush;
            bool ok = test_instance(m);
            std::cout << (ok ? "OK" : "NO") << std::endl << std::flush;
            all_ok &= ok;
        }
        for(int m = 10; m < TESTER_BIG_INSTANCES; m *= 2) {
            std::cout << "Testing with m=" << std::setw(10) << m << ": ";
            bool ok = test_instance(m);
            std::cout << (ok ? "OK" : "NO") << std::endl;
            all_ok &= ok;
        }
        return all_ok;
    }
};

typedef void (*fn)(int INPUT_ARRAY input, int * output, int len); 

class fn_tester : public tester {
public:

    fn_tester(fn reference_fun, fn cuda_fun) 
        : reference_fun(reference_fun), cuda_fun(cuda_fun) { }

private:

    fn reference_fun, cuda_fun;

    bool test_instance(int len) override {

        // generate input
        int * input = utils::random::generate_array<int>(1, 100, len);

        // run reference implementation
        int * reference_output = new int[len];
        reference_fun(input, reference_output, len);

        // run parallel implementation
        int * parallel_output      = new int[len];
        int * parallel_cuda_input  = utils::cuda::allocate_send<int>(input, len);
        int * parallel_cuda_output = utils::cuda::allocate<int>(len);
        cuda_fun(parallel_cuda_input, parallel_cuda_output, len);
        utils::cuda::recv(parallel_output, parallel_cuda_output, len);

        // compare implementations
        bool ok = true; //utils::equals<int>(reference_output, parallel_output, len);
        std::set<int, std::greater<int>> blocks;

        for(int i = 0; i < len; i++) {
            if(reference_output[i] != parallel_output[i]) {
                ok = false;
                std::cout << "Wrong index " << i << " " << reference_output[i] 
                    << " " << parallel_output[i] << std::endl;

                int block = i / SEGSORT_ELEMENTS_PER_BLOCK;
                blocks.insert(block);
            }
        }
        for (std::set<int, std::greater<int>>::iterator itr = blocks.begin(); itr != blocks.end(); ++itr) 
        { 
            int block = *itr; 
            DPRINT_MSG("Error on block %d", block)
            DPRINT_ARR(reference_output + block * SEGSORT_ELEMENTS_PER_BLOCK, SEGSORT_ELEMENTS_PER_BLOCK)
            DPRINT_ARR(parallel_output + block * SEGSORT_ELEMENTS_PER_BLOCK,  SEGSORT_ELEMENTS_PER_BLOCK)
        
        } 

        utils::cuda::deallocate(parallel_cuda_input);
        utils::cuda::deallocate(parallel_cuda_output);
        delete input, reference_output, parallel_output;

        return ok;
    }

};

typedef void (*fn3)(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, float INPUT_ARRAY b_in, float * b_out) ;

class fn3_tester : public tester {

    fn3 reference_fun, cuda_fun;

public:

    fn3_tester(fn3 reference_fun, fn3 cuda_fun) 
        : reference_fun(reference_fun), cuda_fun(cuda_fun) { }

    bool test_instance(int len) override {

        // generate input
        int * input  = utils::random::generate_array<int>(1, 100, len);
        int * a_in   = utils::random::generate_array<int>(1, 100, len);
        float * b_in = utils::random::generate_array<float>(1, 100, len);

        // run reference implementation
        int * reference_output = new int[len];
        int * reference_a_out  = new int[len];
        float * reference_b_out  = new float[len];
        reference_fun(input, reference_output, len, a_in, reference_a_out, b_in, reference_b_out);

        // run parallel implementation
        int * parallel_output      = new int[len];
        int * parallel_a_out       = new int[len];
        float * parallel_b_out     = new float[len];
        int * parallel_cuda_input  = utils::cuda::allocate_send<int>(input, len);
        int * parallel_cuda_a_in   = utils::cuda::allocate_send<int>(a_in, len);
        float * parallel_cuda_b_in = utils::cuda::allocate_send<float>(b_in, len);
        int * parallel_cuda_output = utils::cuda::allocate<int>(len);
        int * parallel_cuda_a_out  = utils::cuda::allocate<int>(len);
        float * parallel_cuda_b_out= utils::cuda::allocate<float>(len);
        cuda_fun(parallel_cuda_input, parallel_cuda_output, len, parallel_cuda_a_in, parallel_cuda_a_out, parallel_cuda_b_in, parallel_cuda_b_out);
        utils::cuda::recv(parallel_output, parallel_cuda_output, len);
        utils::cuda::recv(parallel_a_out,  parallel_cuda_a_out,  len);
        utils::cuda::recv(parallel_b_out,  parallel_cuda_b_out,  len);

        // compare implementations
        bool ok = utils::equals<int>(reference_output, parallel_output, len);

        utils::cuda::deallocate(parallel_cuda_input);
        utils::cuda::deallocate(parallel_cuda_a_in);
        utils::cuda::deallocate(parallel_cuda_b_in);
        utils::cuda::deallocate(parallel_cuda_output);
        utils::cuda::deallocate(parallel_cuda_a_out);
        utils::cuda::deallocate(parallel_cuda_b_out);
        delete input, a_in, b_in; 
        delete reference_output, reference_a_out, reference_b_out;
        delete parallel_output, parallel_a_out, parallel_b_out;

        return ok;
    }
};

class pointer_to_index_tester : public tester {

    bool test_instance(int instance_number) override {

        int m = instance_number;

        // generate input
        int * input = utils::random::generate_array<int>(0, 3, m+1);
        input[m] = 0;
        DPRINT_ARR(input, m+1)
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
        DPRINT_ARR(input, m)
        DPRINT_ARR(reference_output, nnz)
        DPRINT_ARR(parallel_output, nnz)

        utils::cuda::deallocate(parallel_cuda_input);
        utils::cuda::deallocate(parallel_cuda_output);
        delete input, reference_output, parallel_output;

        return ok;
    }
};

class indexes_to_pointers_tester : public tester {

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
};

class algo_transposer_tester : public tester {

    transposers::algo _algo;

public:

    algo_transposer_tester(transposers::algo _algo) : tester(), _algo(_algo) { }

    bool test_sample_instance() {

        int M = 4, N = 6, NNZ = 15;
        int a[] = { 0, 2, 6, 10, 15 };
        int b[] = { 1, 3, 0, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5 };
        float c[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        matrix::SparseMatrix *sm = new matrix::SparseMatrix(M, N, NNZ, matrix::ALL_ZEROS_INITIALIZATION);
        utils::copy_array(sm->csrRowPtr, a, M+1);
        utils::copy_array(sm->csrColIdx, b, NNZ);
        utils::copy_array(sm->csrVal,    c, NNZ);

        matrix::SparseMatrix *sm_refe = new matrix::SparseMatrix(N, M, NNZ, matrix::ALL_ZEROS_INITIALIZATION);
        
        matrix::SparseMatrix *sm_cuda = new matrix::SparseMatrix(N, M, NNZ, matrix::ALL_ZEROS_INITIALIZATION);

        transposers::serial_csr2csc(
            M, N, NNZ,
            sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
            sm_refe->csrRowPtr, sm_refe->csrColIdx, sm_refe->csrVal
        );

        // merge trans implementation
        transposers::cuda_wrapper(
            M, N, NNZ,
            sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
            sm_cuda->csrRowPtr, sm_cuda->csrColIdx, sm_cuda->csrVal,
            _algo
        );

        bool ok =          
            utils::equals(sm_refe->csrRowPtr, sm_cuda->csrRowPtr, N+1) ||
            utils::equals(sm_refe->csrColIdx, sm_cuda->csrColIdx, NNZ) ||
            utils::equals(sm_refe->csrVal, sm_cuda->csrVal, NNZ); 

        delete sm, sm_refe, sm_cuda;

        return ok;

    }

    bool test_instance(int instance_number) {

        int NNZ = instance_number;
        int N = 0, M = 0;
        while(instance_number > N * M) {
            N = utils::random::generate(instance_number*3)+1;
            M = utils::random::generate(instance_number*3)+1;
        }

        DPRINT_MSG("NNZ=%d, M=%d, N=%d", NNZ, M, N)

        //std::cout << "Generating Matrix 0" << std::endl << std::flush;
        matrix::SparseMatrix *sm = new matrix::SparseMatrix(M, N, NNZ);
        //std::cout << "Generating Matrix 1" << std::endl << std::flush;
        matrix::SparseMatrix *sm_refe = new matrix::SparseMatrix(N, M, NNZ, matrix::ALL_ZEROS_INITIALIZATION);
        //std::cout << "Generating Matrix 2" << std::endl << std::flush;
        matrix::SparseMatrix *sm_cuda = new matrix::SparseMatrix(N, M, NNZ, matrix::ALL_ZEROS_INITIALIZATION);

        // reference implementation
        transposers::serial_csr2csc(
            M, N, NNZ,
            sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
            sm_refe->csrRowPtr, sm_refe->csrColIdx, sm_refe->csrVal
        );

        // scan trans implementation
        transposers::cuda_wrapper(
            M, N, NNZ,
            sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
            sm_cuda->csrRowPtr, sm_cuda->csrColIdx, sm_cuda->csrVal,
            _algo
        );

        bool ok =          
            utils::equals(sm_refe->csrRowPtr, sm_cuda->csrRowPtr, N+1) ||
            utils::equals(sm_refe->csrColIdx, sm_cuda->csrColIdx, NNZ) ||
            utils::equals(sm_refe->csrVal, sm_cuda->csrVal, NNZ);

        delete sm, sm_refe, sm_cuda;

        return ok;
    }
};

class matrix_transposer_tester : public tester {

public:
    bool test_instance(int instance_number) {

        bool all_ok = true;

        int NNZ = instance_number;
        int N = 0, M = 0;
        while(instance_number > N * M) {
            N = utils::random::generate(instance_number*3)+1;
            M = utils::random::generate(instance_number*3)+1;
        }

        DPRINT_MSG("M=%d, N=%d, NNZ=%d", M, N, NNZ)

        matrix::FullMatrix * fm   = new matrix::FullMatrix(M, N, NNZ);
        matrix::FullMatrix * fmt  = fm->transpose();
        matrix::SparseMatrix * sm = fm->to_sparse();

        std::cout << std::endl;
        for(int i = matrix::TranspositionMethod::SERIAL; i <= matrix::TranspositionMethod::CUSPARSE2; ++i) {
            matrix::TranspositionMethod tm = static_cast<matrix::TranspositionMethod>(i);
            matrix::SparseMatrix * smt = sm->transpose(tm);
            matrix::FullMatrix * fmtt = new matrix::FullMatrix(smt);
            bool ok = fmt->equals(fmtt);
            all_ok &= ok;
            std::cout << "Transposition method=" << i << " results=" << (ok ? "OK" : "ERR") << std::endl;
            delete smt, fmtt;
        }
        
        delete fm, fmt, sm;

        return all_ok;
    }
};

int old_main(int argc, char **argv) {

    matrix_transposer_tester ta;
    ta.test_many_instances();

    return 0;
}

int main(int argc, char **argv) {

    std::atexit(reinterpret_cast<void(*)()>(cudaDeviceReset));

    indexes_to_pointers_tester  t1;
    pointer_to_index_tester     t2;
    fn_tester                   t3(procedures::reference::scan, procedures::cuda::scan);
    fn_tester                   t4(procedures::reference::segsort, procedures::cuda::segsort);
    fn_tester                   t5(procedures::reference::sort,  procedures::cuda::sort);
    fn3_tester                  t6(procedures::reference::segsort3, procedures::cuda::segsort3);
    fn3_tester                  t7(procedures::reference::sort3, procedures::cuda::sort3);
    algo_transposer_tester      t8(transposers::scan_csr2csc);
    algo_transposer_tester      t9(transposers::merge_csr2csc);
    matrix_transposer_tester    ta;

    tester* tests[] = { &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, &ta };
    std::string names[] = { "IDX TO PTR", "PTR TO IDX", "SCAN", "SEGSORT", "SORT", "SEGSORT3", "SORT3", "SCANTRANS", "MERGETRANS", "MATRIX" };

    bool all_ok = true;
    for(int i = 0; i < 9; i++) {
        std::cout << "\n\n\nTesting " << names[i] << "\n";
        all_ok &= tests[i]->test_many_instances();
    }

    std::cout << "\n\n\nTesting " << names[8-1] << " with example given on paper. Working? ";
    std::cout << (t8.test_sample_instance() ? "YES" : "NO") << std::endl;

    std::cout << "\n\n\nTesting " << names[9-1] << " with example given on paper. Working? ";
    std::cout << (t9.test_sample_instance() ? "YES" : "NO") << std::endl;

    std::cout << "\n\n\nTest result: all is working?" << (all_ok ? "YES" : "NO") << std::endl;

    cudaDeviceReset();
    return 0;
}
