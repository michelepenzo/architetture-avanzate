#include "procedures.hh"

void procedures::cuda::sort(int INPUT_ARRAY input, int * output, int len) {

    // applico prima segsort
    procedures::cuda::segsort(input, output, len);

    // alloco spazio necessario
    int* output_buffer[2] = { output, utils::cuda::allocate<int>(len) };
    int full = 0;

    // applico merge
    for(int BLOCK_SIZE = SEGSORT_ELEMENTS_PER_BLOCK; BLOCK_SIZE < len; BLOCK_SIZE *= 2) {
        
        segmerge_step(output_buffer[full], output_buffer[1-full], len, BLOCK_SIZE);
        full = 1 - full;
    }

    // eventualmente copio nell'array di output nel caso non sia stato l'ultimo
    // ad essere riempito...
    if(full != 0) {
        utils::cuda::copy(output, output_buffer[1], len);
    }

    // dealloco array temporaneo
    utils::cuda::deallocate(output_buffer[1]);
}

void procedures::cuda::sort3(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {

    // applico prima segsort
    procedures::cuda::segsort3(input, output, len, a_in, a_out, b_in, b_out);

    // alloco spazio necessario
    int* output_buffer[2] = { output, utils::cuda::allocate<int>(len) };
    int* a_buffer[2]      = {  a_out, utils::cuda::allocate<int>(len) };
    int* b_buffer[2]      = {  b_out, utils::cuda::allocate<int>(len) };
    int full = 0;

    // applico merge
    for(int BLOCK_SIZE = SEGSORT_ELEMENTS_PER_BLOCK; BLOCK_SIZE < len; BLOCK_SIZE *= 2) {
        
        segmerge3_step(output_buffer[full], output_buffer[1-full], len, BLOCK_SIZE, a_buffer[full], a_buffer[1-full], b_buffer[full], b_buffer[1-full]);
        full = 1 - full;
    }

    // eventualmente copio nell'array di output nel caso non sia stato l'ultimo
    // ad essere riempito...
    if(full != 0) {
        utils::cuda::copy(output, output_buffer[1], len);
        utils::cuda::copy( a_out,      a_buffer[1], len);
        utils::cuda::copy( b_out,      b_buffer[1], len);
    }

    // dealloco array temporaneo
    utils::cuda::deallocate(output_buffer[1]);
    utils::cuda::deallocate(a_buffer[1]);
    utils::cuda::deallocate(b_buffer[1]);
}

void procedures::reference::sort(int INPUT_ARRAY input, int * output, int len) {

    utils::copy_array(output, input, len);
    std::sort(output, output + len);
}

void procedures::reference::sort3(int INPUT_ARRAY input, int * output, int len, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {

    // struttura che implementa il comparatore da applicare sugli array degli indici
    class sort_indices {
    private:
        int const * const mparr;
    public:
        sort_indices(int const * const parr) : mparr(parr) { }
        bool operator()(int i, int j) const { return mparr[i]<mparr[j]; }
    };

    // creo array temporaneo degli indici 0, 1, 2, ..., len-1
    int* indices = new int[len];
    for(int i = 0; i < len; i++) indices[i] = i;

    // ordinamento degli indici, ottengo permutazione degli elementi
    std::sort(indices, indices+len, sort_indices(input));

    // applico la permutazione attraverso gli indici trovati
    for(int j = 0; j < len; j++) {
        output[j] = input[indices[j]];
        a_out[j]  = a_in[indices[j]]; 
        b_out[j]  = b_in[indices[j]];
    }

    // dealloco variabili temporanee
    delete indices;
}