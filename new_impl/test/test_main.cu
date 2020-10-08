#include <iostream>
#include <iomanip>
#include <set>
#include "matrix.hh"
#include "procedures.hh"

typedef void (*fn)(int INPUT_ARRAY input, int * output, int len); 

typedef void (*fn3)(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) ;

bool test_instance(int len, fn3 reference_fun, fn3 cuda_fun){

    // generate input
    int * input = utils::random::generate_array(1, 100, len);
    int * a_in  = input;
    int * b_in  = input;

    // run reference implementation
    int * reference_output = new int[len];
    int * reference_a_out  = new int[len];
    int * reference_b_out  = new int[len];
    reference_fun(input, reference_output, len, a_in, reference_a_out, b_in, reference_b_out);

    // run parallel implementation
    int * parallel_output      = new int[len];
    int * parallel_a_out       = new int[len];
    int * parallel_b_out       = new int[len];
    int * parallel_cuda_input  = utils::cuda::allocate_send<int>(input, len);
    int * parallel_cuda_a_in   = utils::cuda::allocate_send<int>( a_in, len);
    int * parallel_cuda_b_in   = utils::cuda::allocate_send<int>( b_in, len);
    int * parallel_cuda_output = utils::cuda::allocate<int>(len);
    int * parallel_cuda_a_out  = utils::cuda::allocate<int>(len);
    int * parallel_cuda_b_out  = utils::cuda::allocate<int>(len);
    cuda_fun(parallel_cuda_input, parallel_cuda_output, len, parallel_cuda_a_in, parallel_cuda_a_out, parallel_cuda_b_in, parallel_cuda_b_out);
    utils::cuda::recv(parallel_output, parallel_cuda_output, len);
    utils::cuda::recv(parallel_a_out,  parallel_cuda_a_out,  len);
    utils::cuda::recv(parallel_b_out,  parallel_cuda_b_out,  len);

    // compare implementations
    bool ok = true; //utils::equals<int>(reference_output, parallel_output, len);
    for(int i = 0; i < len; i++) {
        if(   reference_output[i] != parallel_output[i]
           || reference_a_out[i]  != parallel_a_out[i]
           || reference_b_out[i]  != parallel_b_out[i]) {
            ok = false;
            std::cout << "Wrong index " << i << ": " 
                << reference_output[i] << " " << parallel_output[i] << "; "
                << reference_a_out[i] << " " << parallel_a_out[i] << "; "
                << reference_b_out[i] << " " << parallel_b_out[i] << "; "
                << std::endl;
        }
    }

    if(!ok) {
        utils::print("input", input, len);
        utils::print("reference_output", reference_output, len);
        utils::print("parallel_output", parallel_output, len);
        utils::print("a_in", a_in, len);
        utils::print("reference_a_out", reference_a_out, len);
        utils::print("parallel_a_out", parallel_a_out, len);
        utils::print("b_in", b_in, len);
        utils::print("reference_b_out", reference_b_out, len);
        utils::print("parallel_b_out", parallel_b_out, len);
    }

    utils::cuda::deallocate(parallel_cuda_input);
    utils::cuda::deallocate(parallel_cuda_a_in);
    utils::cuda::deallocate(parallel_cuda_b_in);
    utils::cuda::deallocate(parallel_cuda_output);
    utils::cuda::deallocate(parallel_cuda_a_out);
    utils::cuda::deallocate(parallel_cuda_b_out);
    delete input; 
    delete reference_output, reference_a_out, reference_b_out;
    delete parallel_output, parallel_a_out, parallel_b_out;

    return ok;
} 

bool test_instance(int len, fn reference_fun, fn cuda_fun) {

    // generate input
    int * input = utils::random::generate_array(1, 4, len);

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

bool test_many_instances(std::string name, fn reference_fun, fn cuda_fun) {
    bool all_ok = true;
    for(int n = 1; n <= 20'000; n++) {
        std::cout << "Testing " << name << " with len=" << std::setw(10) << n << ": ";
        bool ok = test_instance(n, reference_fun, cuda_fun);
        std::cout << (ok ? "OK" : "NO") << std::endl;
        all_ok = all_ok && ok;
    }
    for(int n = 1; n <= 100'000'000; n++) {
        std::cout << "Testing "<< name << " with len=" << std::setw(10) << n << ": ";
        bool ok = test_instance(n, reference_fun, cuda_fun);
        std::cout << (ok ? "OK" : "NO") << std::endl;
        all_ok = all_ok && ok;
        if(n % 1000 == 0) { n *= 2; } // exponential incrementation
    }
    return all_ok;
}

bool test_many_instances(std::string name, fn3 reference_fun, fn3 cuda_fun) {
    bool all_ok = true;
    for(int n = 1; n <= 20'000; n++) {
        std::cout << "Testing " << name << " with len=" << std::setw(10) << n << ":\n";
        bool ok = test_instance(n, reference_fun, cuda_fun);
        std::cout << (ok ? "OK" : "NO") << std::endl;
        all_ok = all_ok && ok;
    }
    for(int n = 1; n <= 100'000'000; n++) {
        std::cout << "Testing "<< name << " with len=" << std::setw(10) << n << ": ";
        bool ok = test_instance(n, reference_fun, cuda_fun);
        std::cout << (ok ? "OK" : "NO") << std::endl;
        all_ok = all_ok && ok;
        if(n % 1000 == 0) { n *= 2; } // exponential incrementation
    }
    return all_ok;
}

bool test_pointer_to_index(int m) {

    // generate input
    int * input = utils::random::generate_array(0, 3, m+1);
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

bool test_many_pointer_to_index() {

    bool all_ok = true;

    for(int m = 1; m < 20'000; m++) {
        std::cout << "Testing pointer_to_index with m=" << std::setw(10) << m << ":\n";
        bool ok = test_pointer_to_index(m);
        std::cout << (ok ? "OK" : "NO") << std::endl;
        all_ok &= ok;
    }
    for(int m = 10; m < 100'000'000; m++) {
        std::cout << "Testing pointer_to_index with m=" << std::setw(10) << m << ": ";
        bool ok = test_pointer_to_index(m);
        std::cout << (ok ? "OK" : "NO") << std::endl;
        if(m % 1000 == 0) { m *= 2; } // exponential incrementation
        all_ok &= ok;
    }
    return all_ok;
}

bool test_single_indexes_to_pointers(int NNZ, int N) {

    //const int NNZ = 15, N = 6;
    //int colIdx[] = {1, 3, 0, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    int * colIdx = utils::random::generate_array(0, N, NNZ);
    int * inter = new int[(HISTOGRAM_BLOCKS+1) * N];
    int * intra = new int[NNZ];
    int * colPtr = new int[N+1]();

    int * colIdx_cuda = utils::cuda::allocate_send<int>(colIdx, NNZ);
    int * inter_cuda = utils::cuda::allocate_zero<int>((HISTOGRAM_BLOCKS+1) * N);
    int * intra_cuda = utils::cuda::allocate_zero<int>(NNZ);
    int * colPtr_cuda = utils::cuda::allocate_zero<int>(N+1);

    int * inter_cuda_out  = new int[(HISTOGRAM_BLOCKS+1) * N];
    int * intra_cuda_out  = new int[NNZ];
    int * colPtr_cuda_out = new int[N+1];

    procedures::reference::indexes_to_pointers(colIdx, NNZ, inter, intra, colPtr, N);

    procedures::cuda::indexes_to_pointers(colIdx_cuda, NNZ, inter_cuda, intra_cuda, colPtr_cuda, N);
    utils::cuda::recv<int>(inter_cuda_out, inter_cuda, (HISTOGRAM_BLOCKS+1) * N);
    utils::cuda::recv<int>(intra_cuda_out, intra_cuda, NNZ);
    utils::cuda::recv<int>(colPtr_cuda_out, colPtr_cuda, N+1);

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

int main(int argc, char **argv) {

    bool all_ok = true;

    all_ok &= test_single_indexes_to_pointers();
    //all_ok &= test_many_pointer_to_index();
    //all_ok &= test_many_instances("scan", procedures::reference::scan, procedures::cuda::scan);
    //all_ok &= test_many_instances("sort", procedures::reference::sort,  procedures::cuda::sort);
    //all_ok &= test_many_instances("sort3", procedures::reference::sort3, procedures::cuda::sort3);
    //all_ok &= test_many_instances("segsort", procedures::reference::segsort, procedures::cuda::segsort);
    //all_ok &= test_many_instances("segsort3", procedures::reference::segsort3, procedures::cuda::segsort3);

    std::cout << "Was test ok: " << (all_ok ? "YES" : "NO") << std::endl;
    return 0;
}

/*

bool procedures::component_test::indexes_to_pointers() {

    const int N = 10000, NNZ = 10000;
    // input
    int *idx = utils::random::generate_array(0, N-1, NNZ);
    DPRINT_ARR(idx, NNZ)

    // reference implementation
    int *ptr = new int[N+1];
    procedures::reference::indexes_to_pointers(idx, NNZ, ptr, N+1);
    DPRINT_ARR(ptr, N+1)

    // cuda implementation
    int *idx_cuda = utils::cuda::allocate_send<int>(idx, NNZ);
    int *ptr_cuda = utils::cuda::allocate_zero<int>(N+1);
    procedures::cuda::indexes_to_pointers(idx_cuda, NNZ, ptr_cuda, N+1);
    int *ptr2 = new int[N+1]; utils::cuda::recv(ptr2, ptr_cuda, N+1);
    DPRINT_ARR(ptr2, N+1)

    bool ok = utils::equals<int>(ptr, ptr2, N+1);

    utils::cuda::deallocate(idx_cuda);
    utils::cuda::deallocate(ptr_cuda);
    delete idx, ptr, ptr2;
    
    return ok;
} 



bool pointers_to_indexes() {

    const int N = 10000, NNZ = 10000;

    int *ptr = utils::random::generate_array(0, 1, N+1);
    ptr[N] = 0;
    utils::prefix_sum(ptr, N+1);
    DPRINT_ARR(ptr, N+1)

    // reference implementation
    int *idx = new int[NNZ];
    reference::pointers_to_indexes(ptr, N+1, idx, NNZ);
    DPRINT_ARR(idx, NNZ)

    // cuda implementation
    int *ptr_cuda = utils::cuda::allocate_send<int>(ptr, N+1);
    int *idx_cuda = utils::cuda::allocate_zero<int>(NNZ);
    procedures::cuda::pointers_to_indexes(ptr_cuda, N+1, idx_cuda, NNZ);
    int *idx2 = new int[N+1]; utils::cuda::recv(idx2, idx_cuda, NNZ);
    DPRINT_ARR(idx2, NNZ)

    bool ok = utils::equals<int>(idx, idx2, NNZ);

    utils::cuda::deallocate(idx_cuda);
    utils::cuda::deallocate(ptr_cuda);
    delete ptr, idx, idx2;
    
    return ok;
}

bool procedures::component_test::segsort() {

    const int N = 10000000;
    // input
    int *arr = utils::random::generate_array(1, 100, N);
    DPRINT_ARR(arr, N)

    // reference implementation
    int *segsort_arr = new int[N];
    procedures::reference::segsort(arr, segsort_arr, N);
    DPRINT_ARR(segsort_arr, N)

    // cuda implementation
    int *segsort_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
    int *segsort_cuda_out = utils::cuda::allocate<int>(N);
    procedures::cuda::segsort(segsort_cuda_in, segsort_cuda_out, N);
    int *segsort_arr_2 = new int[N]; 
    utils::cuda::recv(segsort_arr_2, segsort_cuda_out, N);
    DPRINT_ARR(segsort_arr_2, N)

    bool ok = utils::equals<int>(segsort_arr, segsort_arr_2, N);

    utils::cuda::deallocate(segsort_cuda_in);
    utils::cuda::deallocate(segsort_cuda_out);
    delete arr, segsort_arr, segsort_arr_2;
    
    return ok;
}


#define MIN_RAND_VALUE 0
#define MAX_RAND_VALUE 5000
#define RIPETITION 100
#define BLOCK_SIZE 32
// ===============================================================================
// solo segmerge step
bool procedures::component_test::segmerge() {

    const int N = 10000000;
    // input
    
    bool oks = true;
    //int BLOCK_SIZE = 2;

    for(int j=0; j < RIPETITION; j++){

        int rand_value = utils::random::generate(MIN_RAND_VALUE, MAX_RAND_VALUE);
        int *arr = utils::random::generate_array(1 + rand_value ,10000 + rand_value, N);
        
        DPRINT_ARR(arr, N)

        // reference implementation
        DPRINT_MSG("reference implementation")
        int *segmerge_arr = new int[N];
        procedures::reference::segmerge_step(arr, segmerge_arr, N, BLOCK_SIZE);
        //DPRINT_ARR(segmerge_arr, N)

        // cuda implementation
        DPRINT_MSG("cuda implementation")
        int *segmerge_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_cuda_out = utils::cuda::allocate<int>(N);
        procedures::cuda::segmerge_step(segmerge_cuda_in, segmerge_cuda_out, N, BLOCK_SIZE);
        int *segmerge_arr_2 = new int[N]; 
        utils::cuda::recv(segmerge_arr_2, segmerge_cuda_out, N);
        DPRINT_ARR(segmerge_arr_2, N)

        bool ok = utils::equals<int>(segmerge_arr, segmerge_arr_2, N);
        oks = oks && ok;

        utils::cuda::deallocate(segmerge_cuda_in);
        utils::cuda::deallocate(segmerge_cuda_out);
        delete arr, segmerge_arr, segmerge_arr_2;    
    }
    return oks;
}



// ===============================================================================
// solo segmerge3 step
bool procedures::component_test::segmerge3() {

    const int N = 100000;
    // input
    
    bool oks = true;
    //int BLOCK_SIZE = 2;
    for(int j=0; j < RIPETITION; j++){

        int rand_value = utils::random::generate(MIN_RAND_VALUE, MAX_RAND_VALUE);
        int *arr = utils::random::generate_array(1 + rand_value ,10000 + rand_value, N);
        
        DPRINT_ARR(arr, N)
        // reference implementation
        int *segmerge_arr = new int[N];
        procedures::reference::segmerge_step(arr, segmerge_arr, N, BLOCK_SIZE);
        DPRINT_ARR(segmerge_arr, N)

        // cuda implementation
        int *segmerge_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_a_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_b_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_cuda_out = utils::cuda::allocate<int>(N);
        int *segmerge_a_cuda_out = utils::cuda::allocate<int>(N);
        int *segmerge_b_cuda_out = utils::cuda::allocate<int>(N);
    
        procedures::cuda::segmerge3_step(segmerge_cuda_in, segmerge_cuda_out, N, BLOCK_SIZE,
                                        segmerge_a_cuda_in, segmerge_b_cuda_in, segmerge_a_cuda_out, segmerge_b_cuda_out );

        int *segmerge_arr_2 = new int[N]; 
        utils::cuda::recv(segmerge_arr_2, segmerge_cuda_out, N);
        DPRINT_ARR(segmerge_arr_2, N)

        bool ok = utils::equals<int>(segmerge_arr, segmerge_arr_2, N);
        oks = oks && ok;

        utils::cuda::deallocate(segmerge_cuda_in);
        utils::cuda::deallocate(segmerge_a_cuda_in);
        utils::cuda::deallocate(segmerge_b_cuda_in);
        utils::cuda::deallocate(segmerge_cuda_out);
        utils::cuda::deallocate(segmerge_a_cuda_out);
        utils::cuda::deallocate(segmerge_b_cuda_out);

        delete arr, segmerge_arr, segmerge_arr_2;
    }

    return oks;
}


#define MAX_RAND_VALUE 0
#define MIN_RAND_VALUE 5000
#define RIPETITION 1

// ================================================
//  segmerge sm 
bool procedures::component_test::segmerge_sm() {

    // test with len=10, block size=4
    {
        const int N = 10;
        const int BLOCK_SIZE = 4;
        int * arr = new int[N]{1, 3, 5, 7, 2, 4, 6, 8, 1, 2};
        int * segmerge_sm_arr = new int[N];
        int * segmerge_sm_arr_2 = new int[N]; 
        int * segmerge_sm_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int * segmerge_sm_cuda_out = utils::cuda::allocate<int>(N);
        DPRINT_ARR(arr, N)

        // reference inplementation
        std::cout << "Starting reference implementation...\n";
        procedures::reference::segmerge_sm_step(arr, segmerge_sm_arr, N, BLOCK_SIZE);
        DPRINT_ARR(segmerge_sm_arr, N)

        // cuda implementation
        std::cout << "Starting cuda implementation...\n";
        procedures::cuda::segmerge_sm_step(segmerge_sm_cuda_in, segmerge_sm_cuda_out, N, BLOCK_SIZE);
        utils::cuda::recv(segmerge_sm_arr_2, segmerge_sm_cuda_out, N);
        DPRINT_ARR(segmerge_sm_arr_2, N)

        // check correcness
        bool ok = utils::equals<int>(segmerge_sm_arr, segmerge_sm_arr_2, N);
        if(!ok) {
            std::cout << "Error\n";
            return false;
        } else {
            std::cout << "OK\n";
        }

        // deallocate resources
        utils::cuda::deallocate(segmerge_sm_cuda_in);
        utils::cuda::deallocate(segmerge_sm_cuda_out);
        delete arr, segmerge_sm_arr, segmerge_sm_arr_2;
    }
    

    return true;

    const int N = 100;
    // input

    bool oks = true;
    for(int j=0; j < RIPETITION; j++){

        int rand_value = utils::random::generate(MIN_RAND_VALUE, MAX_RAND_VALUE);
        int *arr = utils::random::generate_array(1 + rand_value, 10000 + rand_value, N);

        DPRINT_ARR(arr, N)

        // reference implementation
        int *segmerge_sm_arr = new int[N];
        procedures::reference::segmerge_sm_step(arr, segmerge_sm_arr, N, BLOCK_SIZE);
        DPRINT_ARR(segmerge_sm_arr, N)

        // cuda implementation
        int *segmerge_sm_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_sm_cuda_out = utils::cuda::allocate<int>(N);
        procedures::cuda::segmerge_sm_step(segmerge_sm_cuda_in, segmerge_sm_cuda_out, N, BLOCK_SIZE);
        int *segmerge_sm_arr_2 = new int[N]; 
        utils::cuda::recv(segmerge_sm_arr_2, segmerge_sm_cuda_out, N);
        DPRINT_ARR(segmerge_sm_arr_2, N)

        bool ok = utils::equals<int>(segmerge_sm_arr, segmerge_sm_arr_2, N);
        oks = oks && ok;



        utils::cuda::deallocate(segmerge_sm_cuda_in);
        utils::cuda::deallocate(segmerge_sm_cuda_out);
        delete arr, segmerge_sm_arr, segmerge_sm_arr_2;


    }
    return oks;
}

bool procedures::component_test::sort() {
    
    bool all_ok = true;

    for(int N = 1000000-1; N < 1000000; N++) {
        // input
        int *arr = utils::random::generate_array(1, 100, N);
        DPRINT_ARR(arr, N)

        // reference implementation
        int *sort_arr = new int[N];
        procedures::reference::sort(arr, sort_arr, N);

        // cuda implementation
        int *sort_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *sort_cuda_out = utils::cuda::allocate<int>(N);
        procedures::cuda::sort(sort_cuda_in, sort_cuda_out, N);
        int *sort_arr_2 = new int[N]; 
        utils::cuda::recv(sort_arr_2, sort_cuda_out, N);

        DPRINT_ARR(sort_arr, N)
        DPRINT_ARR(sort_arr_2, N)
        bool ok = utils::equals<int>(sort_arr, sort_arr_2, N);
        all_ok = all_ok && ok;

        utils::cuda::deallocate(sort_cuda_in);
        utils::cuda::deallocate(sort_cuda_out);
        delete arr, sort_arr, sort_arr_2;
    }

    std::cout << "All ok: " << all_ok << std::endl;
    
    return all_ok;
}

// ================================================
//  segmerge3 sm 
bool procedures::component_test::segmerge3_sm() {

    const int N = 10;
    const int BLOCK_SIZE = 4;
    // input

    bool oks = true;

    for(int j=0; j < RIPETITION; j++){
        int rand_value = utils::random::generate(MIN_RAND_VALUE, MAX_RAND_VALUE);
        int *arr = utils::random::generate_array(1, 6, N);

        DPRINT_ARR(arr, N)

        // reference implementation
        int *segmerge_sm_arr = new int[N];
        int *segmerge_a_in_arr = new int[N];
        int *segmerge_a_out_arr = new int[N];
        int *segmerge_b_in_arr = new int[N];
        int *segmerge_b_out_arr = new int[N];
        procedures::reference::segmerge3_sm_step(arr, segmerge_sm_arr, N, BLOCK_SIZE,
                                                segmerge_a_in_arr, segmerge_a_out_arr, segmerge_b_in_arr, segmerge_b_out_arr);

        DPRINT_ARR(segmerge_sm_arr, N)

        // cuda implementation
        int *segmerge_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_a_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_b_cuda_in  = utils::cuda::allocate_send<int>(arr, N);
        int *segmerge_cuda_out = utils::cuda::allocate<int>(N);
        int *segmerge_a_cuda_out = utils::cuda::allocate<int>(N);
        int *segmerge_b_cuda_out = utils::cuda::allocate<int>(N);
    
        procedures::cuda::segmerge3_sm_step(segmerge_cuda_in, segmerge_cuda_out, N, BLOCK_SIZE,
                                            segmerge_a_cuda_in, segmerge_b_cuda_in, segmerge_a_cuda_out, segmerge_b_cuda_out);
        int *segmerge_sm_arr_2 = new int[N]; 
        utils::cuda::recv(segmerge_sm_arr_2, segmerge_cuda_out, N);
        DPRINT_ARR(segmerge_sm_arr_2, N)

        bool ok = utils::equals<int>(segmerge_sm_arr, segmerge_sm_arr_2, N);
        oks = oks && ok;
        utils::cuda::deallocate(segmerge_cuda_in);
        utils::cuda::deallocate(segmerge_a_cuda_in);
        utils::cuda::deallocate(segmerge_b_cuda_in);
        utils::cuda::deallocate(segmerge_cuda_out);
        utils::cuda::deallocate(segmerge_a_cuda_out);
        utils::cuda::deallocate(segmerge_b_cuda_out);
        delete arr, segmerge_sm_arr, segmerge_sm_arr_2, segmerge_a_in_arr, segmerge_a_out_arr, segmerge_b_in_arr, segmerge_b_out_arr;

    }
    return oks;
}

*/