#include "transposer.hh"

__device__
void ___copy(int * output, int INPUT_ARRAY input, int len) {
    for(int i = 0; i < len; i++) {
        output[i] = input[i];
    }
}

__global__
void segmerge_step_kernel(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {

    int couple_block_id = blockIdx.x;
    int start_1 = 2 * couple_block_id * BLOCK_SIZE;
    int start_2 = (2 * couple_block_id + 1) * BLOCK_SIZE;
    int end_1 = min((2 * couple_block_id + 1) * BLOCK_SIZE, len);
    int end_2 = min((2 * couple_block_id + 2) * BLOCK_SIZE, len);

    int current_1 = start_1;
    int current_2 = start_2;
    int current_output = start_1;
    
    // merge
    while(current_1 < end_1 && current_2 < end_2) {
        if(input[current_1] <= input[current_2]) {
            output[current_output] = input[current_1];
            current_1++;
        } else {
            output[current_output] = input[current_2];
            current_2++;
        }
        current_output++;
    }

    // finisco le rimanenze del primo blocco
    ___copy(output + current_output, input + current_1, end_1 - current_1);

    // finisco le rimanenze del secondo blocco
    ___copy(output + current_output, input + current_2, end_2 - current_2);
}

__global__
void segmerge3_step_kernel(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {

    int couple_block_id = blockIdx.x;
    int start_1 = 2 * couple_block_id * BLOCK_SIZE;
    int start_2 = (2 * couple_block_id + 1) * BLOCK_SIZE;
    int end_1 = min((2 * couple_block_id + 1) * BLOCK_SIZE, len);
    int end_2 = min((2 * couple_block_id + 2) * BLOCK_SIZE, len);

    int current_1 = start_1;
    int current_2 = start_2;
    int current_output = start_1;
    
    // merge
    while(current_1 < end_1 && current_2 < end_2) {
        if(input[current_1] <= input[current_2]) {
            output[current_output] = input[current_1];
            a_out[current_output]  = a_in[current_1];
            b_out[current_output]  = b_in[current_1];
            current_1++;
        } else {
            output[current_output] = input[current_2];
            a_out[current_output]  = a_in[current_2];
            b_out[current_output]  = b_in[current_2];
            current_2++;
        }
        current_output++;
    }

    // finisco le rimanenze del primo blocco
    ___copy(output + current_output, input + current_1, end_1 - current_1);
    ___copy(a_out  + current_output, a_in  + current_1, end_1 - current_1);
    ___copy(b_out  + current_output, b_in  + current_1, end_1 - current_1);

    // finisco le rimanenze del secondo blocco
    ___copy(output + current_output, input + current_2, end_2 - current_2);
    ___copy(a_out  + current_output, a_in  + current_2, end_2 - current_2);
    ___copy(b_out  + current_output, b_in  + current_2, end_2 - current_2);
}

void transposer::cuda::segmerge_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {

    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
    segmerge_step_kernel<<<DIV_THEN_CEIL(BLOCK_NUMBER, 2), 1>>>(input, output, len, BLOCK_SIZE);
    CUDA_CHECK_ERROR
}

void transposer::cuda::segmerge3_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {

    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
    segmerge3_step_kernel<<<DIV_THEN_CEIL(BLOCK_NUMBER, 2), 1>>>(input, output, len, BLOCK_SIZE, a_in, a_out, b_in, b_out);
    CUDA_CHECK_ERROR
}

void transposer::reference::segmerge_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE) {
    segmerge3_step(input, output, len, BLOCK_SIZE, NULL, NULL, NULL, NULL);
}

void transposer::reference::segmerge3_step(int INPUT_ARRAY input, int * output, int len, int BLOCK_SIZE, int INPUT_ARRAY a_in, int * a_out, int INPUT_ARRAY b_in, int * b_out) {

    bool all_three = a_in != NULL && a_out != NULL && b_in != NULL && b_out != NULL;
    const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);

    for(int couple_block_id = 0; couple_block_id < DIV_THEN_CEIL(BLOCK_NUMBER, 2); couple_block_id++) {

        int start_1 = 2 * couple_block_id * BLOCK_SIZE;
        int start_2 = (2 * couple_block_id + 1) * BLOCK_SIZE;
        int end_1 = min((2 * couple_block_id + 1) * BLOCK_SIZE, len);
        int end_2 = min((2 * couple_block_id + 2) * BLOCK_SIZE, len);

        int current_1 = start_1;
        int current_2 = start_2;
        int current_output = start_1;
        
        // merge
        while(current_1 < end_1 && current_2 < end_2) {
            if(input[current_1] <= input[current_2]) {
                output[current_output] = input[current_1];
                if(all_three) a_out[current_output]  = a_in[current_1];
                if(all_three) b_out[current_output]  = b_in[current_1];
                current_1++;
            } else {
                output[current_output] = input[current_2];
                if(all_three) a_out[current_output]  = a_in[current_2];
                if(all_three) b_out[current_output]  = b_in[current_2];
                current_2++;
            }
            current_output++;
        }

        // finisco le rimanenze del primo blocco
        utils::copy_array<int>(output + current_output, input + current_1, end_1 - current_1);
        if(all_three) utils::copy_array<int>(a_out + current_output, a_in + current_1, end_1 - current_1);
        if(all_three) utils::copy_array<int>(b_out + current_output, b_in + current_1, end_1 - current_1);

        // finisco le rimanenze del secondo blocco
        utils::copy_array<int>(output + current_output, input + current_2, end_2 - current_2);
        if(all_three) utils::copy_array<int>(a_out + current_output, a_in + current_2, end_2 - current_2);
        if(all_three) utils::copy_array<int>(b_out + current_output, b_in + current_2, end_2 - current_2);
    }
}

