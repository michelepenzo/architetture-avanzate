#pragma once
#include "procedures.hh"

#ifndef MERGE_STEP_HH
#define MERGE_STEP_HH

#define MERGE_SMALL_MAX_BLOCK_SIZE 256
#define MERGE3_SMALL_MAX_BLOCK_SIZE 256
#define MERGE_BIG_SPLITTER_DISTANCE 128

namespace procedures {

    namespace cuda {

        NUMERIC_TEMPLATE(T)
        __device__ inline int find_position_in_sorted_array(T element_to_search, T INPUT_ARRAY input, int len) {

            if(len <= 0) {
                return 0;
            }

            int start = 0;
            int end = start + len;

            while(start < end) {
                int current = (start + end) / 2;
                if(input[current] < element_to_search) {
                    start = current + 1;
                } else if(input[current] >= element_to_search) {
                    end = current;
                }
            }

            return start;
        }

        NUMERIC_TEMPLATE(T)
        __device__ inline int find_last_position_in_sorted_array(T element_to_search, T INPUT_ARRAY input, int len) {

            int index = find_position_in_sorted_array(element_to_search, input, len);
            while(input[index] == element_to_search && index < len) {
                //printf("input[%d] = %d == %d, index++\n", index, input[index], element_to_search);
                index++;
            }
            return index;
        }

        inline int calculate_splitter_number(int len, int BLOCK_SIZE) {

            int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);
            int COUPLE_OF_BLOCKS = BLOCK_NUMBER / 2;
            int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_BIG_SPLITTER_DISTANCE);
            
            if((BLOCK_NUMBER%2 == 1) || (len % BLOCK_SIZE == 0)) {
                // se il numero dei blocchi è dispari allora tutte le coppie hanno
                // un numero uguale di splitter pari al massimo
                //     oppure
                // se stranamente la lunghezza è multipla del distanziatore di splitter
                return 2 * COUPLE_OF_BLOCKS * SPLITTER_PER_BLOCKS;

            } else {
                // altrimenti l'ultimo elemento dell'ultima coppia ha meno splitters
                int rem = len % BLOCK_SIZE;
                int SPLITTER_PER_LAST_BLOCK = DIV_THEN_CEIL(rem, MERGE_BIG_SPLITTER_DISTANCE);
            
                int all_but_last_couple = 2 * (COUPLE_OF_BLOCKS-1) * SPLITTER_PER_BLOCKS;
                int last_couple = SPLITTER_PER_BLOCKS + SPLITTER_PER_LAST_BLOCK;
                return all_but_last_couple + last_couple;
            }
        }

        // =================================================================
        // MERGE SMALL KERNELS =============================================
        // =================================================================

        NUMERIC_TEMPLATE(T)
        __global__ void merge_small_step_kernel(T INPUT_ARRAY input, T * output, int len, int BLOCK_SIZE) {

            __shared__ T temp_a_in[MERGE_SMALL_MAX_BLOCK_SIZE];
            __shared__ T temp_b_in[MERGE_SMALL_MAX_BLOCK_SIZE];
            __shared__ T temp_out[2*MERGE_SMALL_MAX_BLOCK_SIZE];

            // una griglia di thread ogni due blocchi
            const int COUPLE_BLOCK_ID = blockIdx.x;
            const int i = threadIdx.x;

            // caricamento del blocco a sinistra della coppia
            const int START_A = 2 * COUPLE_BLOCK_ID * BLOCK_SIZE;
            const int END_A = min((2 * COUPLE_BLOCK_ID + 1) * BLOCK_SIZE, len);
            const int LEN_A = max(0, END_A - START_A);
            if(i < LEN_A) {
                temp_a_in[i] = input[START_A + i];
            }
            // caricamento del blocco a destra della coppia
            const int START_B = (2 * COUPLE_BLOCK_ID + 1) * BLOCK_SIZE;
            const int END_B = min((2 * COUPLE_BLOCK_ID + 2) * BLOCK_SIZE, len);
            const int LEN_B = max(0, END_B - START_B);
            if(i < LEN_B) {
                temp_b_in[i] = input[START_B + i];
            }
            // aspetto che i blocchi 'temp_*_in' siano stati caricati completamente
            __syncthreads();

            // recupero le posizioni dell'i-esimo elemento di A e B
            if(i < LEN_A) {
                int k = i + find_position_in_sorted_array(temp_a_in[i], temp_b_in, LEN_B);
                temp_out[k] = temp_a_in[i];
            }
            if(i < LEN_B) {
                int k = i + find_last_position_in_sorted_array(temp_b_in[i], temp_a_in, LEN_A);
                temp_out[k] = temp_b_in[i];
            }
            // aspetto che il blocco temp_out sia stati caricato completamente
            __syncthreads();
            //if(i == 0) {
            //    printf("(%2d):\n"
            //        "temp_a = [%2d %2d %2d %2d]\n"
            //        "temp_b = [%2d %2d %2d %2d]\n"
            //        "temp_out = [%2d %2d %2d %2d %2d %2d %2d %2d]\n",
            //        COUPLE_BLOCK_ID,
            //        temp_a_in[0], temp_a_in[1], temp_a_in[2], temp_a_in[3], 
            //        temp_b_in[0], temp_b_in[1], temp_b_in[2], temp_b_in[3], 
            //        temp_out[0], temp_out[1], temp_out[2], temp_out[3], 
            //        temp_out[4], temp_out[5], temp_out[6], temp_out[7]
            //    );
            //}
            
            // salvo il dato in output
            const int START_OUTPUT = START_A;
            if(2*i < LEN_A + LEN_B) {
                output[START_OUTPUT + 2*i] = temp_out[2*i];
            }
            if(2*i+1 < LEN_A + LEN_B) {
                output[START_OUTPUT + 2*i+1] = temp_out[2*i+1];
            }
        }

        NUMERIC_TEMPLATE3(T, T2, T3)
        __global__ void merge3_small_step_kernel(T INPUT_ARRAY input, T * output, T2 INPUT_ARRAY val1_input, T2 * val1_output, T3 INPUT_ARRAY val2_input, T3 * val2_output, int len, int BLOCK_SIZE) {

            __shared__ T temp_a_in[MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T temp_b_in[MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T temp_out[2*MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T2 val1_temp_a_in[MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T2 val1_temp_b_in[MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T2 val1_temp_out[2*MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T3 val2_temp_a_in[MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T3 val2_temp_b_in[MERGE3_SMALL_MAX_BLOCK_SIZE];
            __shared__ T3 val2_temp_out[2*MERGE3_SMALL_MAX_BLOCK_SIZE];

            // una griglia di thread ogni due blocchi
            const int COUPLE_BLOCK_ID = blockIdx.x;
            const int i = threadIdx.x;

            // caricamento del blocco a sinistra della coppia
            const int START_A = 2 * COUPLE_BLOCK_ID * BLOCK_SIZE;
            const int END_A = min((2 * COUPLE_BLOCK_ID + 1) * BLOCK_SIZE, len);
            const int LEN_A = max(0, END_A - START_A);
            if(i < LEN_A) {
                temp_a_in[i] = input[START_A + i];
                val1_temp_a_in[i] = val1_input[START_A + i];
                val2_temp_a_in[i] = val2_input[START_A + i];
            }
            // caricamento del blocco a destra della coppia
            const int START_B = (2 * COUPLE_BLOCK_ID + 1) * BLOCK_SIZE;
            const int END_B = min((2 * COUPLE_BLOCK_ID + 2) * BLOCK_SIZE, len);
            const int LEN_B = max(0, END_B - START_B);
            if(i < LEN_B) {
                temp_b_in[i] = input[START_B + i];
                val1_temp_b_in[i] = val1_input[START_B + i];
                val2_temp_b_in[i] = val2_input[START_B + i];
            }
            // aspetto che i blocchi 'temp_*_in' siano stati caricati completamente
            __syncthreads();

            // recupero le posizioni dell'i-esimo elemento di A e B
            if(i < LEN_A) {
                int k = i + find_position_in_sorted_array(temp_a_in[i], temp_b_in, LEN_B);
                temp_out[k] = temp_a_in[i];
                val1_temp_out[k] = val1_temp_a_in[i];
                val2_temp_out[k] = val2_temp_a_in[i];
            }
            if(i < LEN_B) {
                int k = i + find_last_position_in_sorted_array(temp_b_in[i], temp_a_in, LEN_A);
                temp_out[k] = temp_b_in[i];
                val1_temp_out[k] = val1_temp_b_in[i];
                val2_temp_out[k] = val2_temp_b_in[i];
            }
            // aspetto che il blocco temp_out sia stati caricato completamente
            __syncthreads();
            
            // salvo il dato in output
            const int START_OUTPUT = START_A;
            if(2*i < LEN_A + LEN_B) {
                output[START_OUTPUT + 2*i] = temp_out[2*i];
                val1_output[START_OUTPUT + 2*i] = val1_temp_out[2*i];
                val2_output[START_OUTPUT + 2*i] = val2_temp_out[2*i];
                if(START_OUTPUT + 2*i > len) {
                    printf("!!!!Error!!!! %d > %d\n", START_OUTPUT + 2*i, len);
                }
            }
            if(2*i+1 < LEN_A + LEN_B) {
                output[START_OUTPUT + 2*i+1] = temp_out[2*i+1];
                val1_output[START_OUTPUT + 2*i+1] = val1_temp_out[2*i+1];
                val2_output[START_OUTPUT + 2*i+1] = val2_temp_out[2*i+1];
                if(START_OUTPUT + 2*i +1 > len) {
                    printf("!!!!Error!!!! %d > %d\n", START_OUTPUT + 2*i, len);
                }
            }
        }

        // =================================================================
        // MERGE BIG KERNELS ===============================================
        // =================================================================

        NUMERIC_TEMPLATE(T)
        __global__ inline void generate_splitter_kernel(T INPUT_ARRAY input, T * splitter, int * indexA, int * indexB, int len, int BLOCK_SIZE) {

            const int couple_block_id = blockIdx.x;
            const int thid = threadIdx.x;

            // entrambi inputA, inputB esistono (eventualmente B ha lunghezza < BLOCK_SIZE se ultimo blocco)
            T * inputA = input + 2 * couple_block_id * BLOCK_SIZE;
            T * inputB = input + (2 * couple_block_id + 1) * BLOCK_SIZE;
            int lenA = BLOCK_SIZE;
            int lenB = min((2 * couple_block_id + 2) * BLOCK_SIZE, len) - (2 * couple_block_id + 1) * BLOCK_SIZE;

            // mi sposto verso gli indici corretti di splitter, indexA, indexB
            const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_BIG_SPLITTER_DISTANCE);
            splitter = splitter + 2 * couple_block_id * SPLITTER_PER_BLOCKS;
            indexA = indexA + 2 * couple_block_id * SPLITTER_PER_BLOCKS;
            indexB = indexB + 2 * couple_block_id * SPLITTER_PER_BLOCKS;

            // riempio gli elementi
            int i;
            for(int i = thid; i < SPLITTER_PER_BLOCKS; i += blockDim.x) {
                splitter[i] = inputA[i*MERGE_BIG_SPLITTER_DISTANCE];
                indexA[i] = i*MERGE_BIG_SPLITTER_DISTANCE;
                indexB[i] = find_position_in_sorted_array(splitter[i], inputB, lenB);
            }
            __syncthreads();

            for(i = thid; i < SPLITTER_PER_BLOCKS && i*MERGE_BIG_SPLITTER_DISTANCE < lenB; i += blockDim.x) {
                T element = inputB[i*MERGE_BIG_SPLITTER_DISTANCE];
                // save splitter
                splitter[SPLITTER_PER_BLOCKS + i] = element;
                indexA[SPLITTER_PER_BLOCKS + i] = find_last_position_in_sorted_array(element, inputA, lenA);
                indexB[SPLITTER_PER_BLOCKS + i] = i*MERGE_BIG_SPLITTER_DISTANCE;
                //printf("(%2d): element %d from B is position %d of A\n", couple_block_id, element, indexA[SPLITTER_PER_BLOCKS + i]);
            }
        }

        __global__ inline void fix_indexes_kernel(int * indexA, int * indexB, int len, int BLOCK_SIZE, int SPLITTER_NUMBER, int SPLITTER_PER_BLOCK) {

            int couple_block_id = blockIdx.x;

            if(threadIdx.x == 0) {

                // calcolo gli indici di inizio e fine degli splitter che devo processare
                int startSplitter = 2 * couple_block_id * SPLITTER_PER_BLOCK;
                int endSplitter  = min(2 * (couple_block_id + 1) * SPLITTER_PER_BLOCK, SPLITTER_NUMBER);
                int lenSplitter = endSplitter - startSplitter;
                // la lunghezza di A è sempre BLOCK_SIZE, la lunghezza di B?... dipende se è l'ultimo
                int lenB = min(2 * (couple_block_id + 1) * BLOCK_SIZE, len) - ((2 * couple_block_id + 1) * BLOCK_SIZE);
                
                // printf("(%2d) split[%3d:%3d] {max %d}\n", blockIdx.x, startSplitter, endSplitter, SPLITTER_NUMBER);
            
                for(int i = 0; i < lenSplitter - 1; i += 1) {
                    indexA[startSplitter+i] = indexA[startSplitter+i+1];
                    indexB[startSplitter+i] = indexB[startSplitter+i+1];
                }

                // l'ultimo elemento contiene la dimensione del blocco
                indexA[startSplitter+lenSplitter-1] = BLOCK_SIZE;
                indexB[startSplitter+lenSplitter-1] = lenB;
                __syncthreads();
            }
        }

        NUMERIC_TEMPLATE(T)
        __global__ inline void uniform_merge_kernel(T INPUT_ARRAY input, T * output, int INPUT_ARRAY indexA, int INPUT_ARRAY indexB, int len, int BLOCK_SIZE) {

            __shared__ T temp_in[2 * MERGE_BIG_SPLITTER_DISTANCE];
            __shared__ T temp_out[2 * MERGE_BIG_SPLITTER_DISTANCE];

            for(int i = 0; i < 2 * MERGE_BIG_SPLITTER_DISTANCE; i++) {
                temp_in[i] = 0;
                temp_out[i] = 0;
            }

            // processa l'elemento dello splitter
            const int splitter_index = blockIdx.x;

            // recupera blocco sul quale stai lavorando
            const int SPLITTER_PER_BLOCK = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_BIG_SPLITTER_DISTANCE);
            int couple_block_id = splitter_index / (2 * SPLITTER_PER_BLOCK);
            int item = splitter_index % (2 * SPLITTER_PER_BLOCK);

            // recupera estremi sui quali lavorare
            T * inputA = input + 2 * couple_block_id * BLOCK_SIZE;
            T * inputB = input + (2 * couple_block_id + 1) * BLOCK_SIZE;
            int startA = (item == 0) ? 0 : indexA[splitter_index-1];
            int startB = (item == 0) ? 0 : indexB[splitter_index-1];
            int endA   = indexA[splitter_index];
            int endB   = indexB[splitter_index];
            
            // carico gli elementi in temp_in
            if(endA - startA > MERGE_BIG_SPLITTER_DISTANCE) printf("!!!Error A[%d:%d] > %d SM SP S\n", startA, endA, MERGE_BIG_SPLITTER_DISTANCE);
            utils::cuda::devcopy<T>(temp_in, inputA + startA, endA - startA);
            if(endB - startB > MERGE_BIG_SPLITTER_DISTANCE) printf("!!!Error B[%d:%d] > %d SM SP S\n", startB, endB, MERGE_BIG_SPLITTER_DISTANCE);
            utils::cuda::devcopy<T>(temp_in + MERGE_BIG_SPLITTER_DISTANCE, inputB + startB, endB - startB);
            __syncthreads();

            // effettuo merge
            for(int i = 0; i < endA - startA; i++) {
                T element = temp_in[i];
                int posInA = i;
                int posInB = find_position_in_sorted_array(element, temp_in + MERGE_BIG_SPLITTER_DISTANCE, endB - startB);
                int k = posInA + posInB;
                temp_out[k] = element;
            }
            __syncthreads();

            for(int i = 0; i < endB - startB; i++) {
                T element = temp_in[MERGE_BIG_SPLITTER_DISTANCE + i];
                int k = i + find_last_position_in_sorted_array(element, temp_in, endA - startA);
                temp_out[k] = element;
            }
            __syncthreads();

            // salva output
            output = output + 2 * couple_block_id * BLOCK_SIZE;
            for(int i = 0; i < (endA - startA) + (endB - startB); i++) {
                output[startA + startB + i] = temp_out[i];
            }
        }

        NUMERIC_TEMPLATE3(T, T2, T3)
        __global__ inline void uniform_merge3_kernel(T INPUT_ARRAY input, T * output, int INPUT_ARRAY indexA, int INPUT_ARRAY indexB, T2 INPUT_ARRAY a_in, T2 * a_out, T3 INPUT_ARRAY b_in, T3 * b_out, int len, int BLOCK_SIZE) {

            __shared__ T temp_in[2 * MERGE_BIG_SPLITTER_DISTANCE];
            __shared__ T temp_out[2 * MERGE_BIG_SPLITTER_DISTANCE];
            __shared__ T2 temp_a_in[2 * MERGE_BIG_SPLITTER_DISTANCE];
            __shared__ T2 temp_a_out[2 * MERGE_BIG_SPLITTER_DISTANCE];
            __shared__ T3 temp_b_in[2 * MERGE_BIG_SPLITTER_DISTANCE];
            __shared__ T3 temp_b_out[2 * MERGE_BIG_SPLITTER_DISTANCE];

            for(int i = 0; i < 2 * MERGE_BIG_SPLITTER_DISTANCE; i++) {
                temp_in[i] = 0;
                temp_out[i] = 0;
                temp_a_in[i] = 0;
                temp_a_out[i] = 0;
                temp_b_in[i] = 0;
                temp_b_out[i] = 0;
            }

            // processa l'elemento dello splitter
            const int splitter_index = blockIdx.x;

            // recupera blocco sul quale stai lavorando
            const int SPLITTER_PER_BLOCK = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_BIG_SPLITTER_DISTANCE);
            int couple_block_id = splitter_index / (2 * SPLITTER_PER_BLOCK);
            int item = splitter_index % (2 * SPLITTER_PER_BLOCK);

            // recupera estremi sui quali lavorare
            int startA = (item == 0) ? 0 : indexA[splitter_index-1];
            int startB = (item == 0) ? 0 : indexB[splitter_index-1];
            int endA   = indexA[splitter_index];
            int endB   = indexB[splitter_index];    
            if(endA - startA > MERGE_BIG_SPLITTER_DISTANCE) {
                printf("ERROR i=%2d BLK_SIZE=%2d - "
                       "A[%d:%d] > %d SM SP S\n",
                        splitter_index, BLOCK_SIZE,
                        startA, endA, 
                        MERGE_BIG_SPLITTER_DISTANCE
                );
            }
            if(endB - startB > MERGE_BIG_SPLITTER_DISTANCE) {
                printf("ERROR i=%2d BLK_SIZE=%2d - "
                       "B[%d:%d] > %d SM SP S\n",
                        splitter_index, BLOCK_SIZE,
                        startB, endB, 
                        MERGE_BIG_SPLITTER_DISTANCE
                );
            }
            
            // carico gli elementi in temp_in
            int OFFSET_A = 2*couple_block_id*BLOCK_SIZE;
            int OFFSET_B = (2*couple_block_id+1)*BLOCK_SIZE;
            utils::cuda::devcopy<T>(temp_in, 
                input + OFFSET_A + startA, endA - startA);
            utils::cuda::devcopy<T>(temp_in + MERGE_BIG_SPLITTER_DISTANCE, 
                input + OFFSET_B + startB, endB - startB);

            utils::cuda::devcopy<T2>(temp_a_in, 
                a_in  + OFFSET_A + startA, endA - startA);
            utils::cuda::devcopy<T2>(temp_a_in + MERGE_BIG_SPLITTER_DISTANCE, 
                a_in  + OFFSET_B + startB, endB - startB);

            utils::cuda::devcopy<T3>(temp_b_in, 
                b_in  + OFFSET_A + startA, endA - startA);
            utils::cuda::devcopy<T3>(temp_b_in + MERGE_BIG_SPLITTER_DISTANCE, 
                b_in  + OFFSET_B + startB, endB - startB);

            //printf("(%2d): COUPLE=%2d, OFFSET_A=%2d, OFFSET_B=%2d, A[%2d:%2d] B[%2d:%2d]\n"
            //       "temp__ = [%2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d]\n"
            //       "temp_a = [%2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d]\n"
            //       "temp_b = [%2.0f, %2.0f, %2.0f, %2.0f, %2.0f, %2.0f, %2.0f, %2.0f]\n",
            //    splitter_index,
            //    couple_block_id, OFFSET_A, OFFSET_B,
            //    startA, endA, startB, endB,
            //    temp_in[0], temp_in[1], temp_in[2], temp_in[3],
            //    temp_in[4], temp_in[5], temp_in[6], temp_in[7],
            //    temp_a_in[0], temp_a_in[1], temp_a_in[2], temp_a_in[3],
            //    temp_a_in[4], temp_a_in[5], temp_a_in[6], temp_a_in[7],
            //    temp_b_in[0], temp_b_in[1], temp_b_in[2], temp_b_in[3],
            //    temp_b_in[4], temp_b_in[5], temp_b_in[6], temp_b_in[7]
            //);

            // effettuo merge
            for(int i = 0; i < endA - startA; i++) {
                T  element = temp_in[i];
                T2 elementA = temp_a_in[i];
                T3 elementB = temp_b_in[i];
                int posInA = i;
                int posInB = find_position_in_sorted_array(element, temp_in + MERGE_BIG_SPLITTER_DISTANCE, endB - startB);
                int k = posInA + posInB;
                temp_out[k] = element;
                temp_a_out[k] = elementA;
                temp_b_out[k] = elementB;
            }
            __syncthreads();

            for(int i = 0; i < endB - startB; i++) {
                T  element = temp_in[MERGE_BIG_SPLITTER_DISTANCE + i];
                T2 elementA = temp_a_in[MERGE_BIG_SPLITTER_DISTANCE + i];
                T3 elementB = temp_b_in[MERGE_BIG_SPLITTER_DISTANCE + i];
                int k = i + find_last_position_in_sorted_array(element, temp_in, endA - startA);
                temp_out[k] = element;
                temp_a_out[k] = elementA;
                temp_b_out[k] = elementB;
            }
            __syncthreads();

            //printf("OUT\n");
            //printf("(%2d): COUPLE=%2d, OFFSET_A=%2d, OFFSET_B=%2d, A[%2d:%2d] B[%2d:%2d]\n"
            //       "temp__ = [%2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d]\n"
            //       "temp_a = [%2d, %2d, %2d, %2d, %2d, %2d, %2d, %2d]\n"
            //       "temp_b = [%2.0f, %2.0f, %2.0f, %2.0f, %2.0f, %2.0f, %2.0f, %2.0f]\n",
            //    splitter_index,
            //    couple_block_id, OFFSET_A, OFFSET_B,
            //    startA, endA, startB, endB,
            //    temp_out[0], temp_out[1], temp_out[2], temp_out[3],
            //    temp_out[4], temp_out[5], temp_out[6], temp_out[7],
            //    temp_a_out[0], temp_a_out[1], temp_a_out[2], temp_a_out[3],
            //    temp_a_out[4], temp_a_out[5], temp_a_out[6], temp_a_out[7],
            //    temp_b_out[0], temp_b_out[1], temp_b_out[2], temp_b_out[3],
            //    temp_b_out[4], temp_b_out[5], temp_b_out[6], temp_b_out[7]
            //);

            // salva output
            for(int i = 0; i < (endA - startA) + (endB - startB); i++) {
                output[OFFSET_A + startA + startB + i] = temp_out[i];
                a_out [OFFSET_A + startA + startB + i] = temp_a_out[i];
                b_out [OFFSET_A + startA + startB + i] = temp_b_out[i];
                if(OFFSET_A + startA + startB + i > len) {
                    printf("!!!!!Error %d > %d\n", OFFSET_A + startA + startB + i, len);
                }
            }
        }

        // =================================================================
        // INTERFACE =======================================================
        // =================================================================

        NUMERIC_TEMPLATE3(T, T2, T3)
        inline void merge3_step(T INPUT_ARRAY input, T * output, T2 INPUT_ARRAY val1_input, T2 * val1_output, T3 INPUT_ARRAY val2_input, T3 * val2_output, int len, int BLOCK_SIZE) {
            
            const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);

            if(BLOCK_SIZE <= MERGE_SMALL_MAX_BLOCK_SIZE) {
                DPRINT_MSG("Block size is %d: calling MERGE SMALL", BLOCK_SIZE)
                merge3_small_step_kernel<T, T2, T3><<<DIV_THEN_CEIL(BLOCK_NUMBER, 2), MERGE_SMALL_MAX_BLOCK_SIZE>>>(input, output, val1_input, val1_output, val2_input, val2_output, len, BLOCK_SIZE);
            
            } else {
                DPRINT_MSG("Block size is %d: calling MERGE BIG", BLOCK_SIZE)
                const int COUPLE_OF_BLOCKS = DIV_THEN_CEIL(len, BLOCK_SIZE) / 2;
                const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_BIG_SPLITTER_DISTANCE);
                const int SPLITTER_NUMBER = calculate_splitter_number(len, BLOCK_SIZE);

                if(COUPLE_OF_BLOCKS > 0) {
                    // 1. genera splitter
                    T * splitter = utils::cuda::allocate<T>(SPLITTER_NUMBER);
                    int * indexA = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                    int * indexB = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                    generate_splitter_kernel<T><<<COUPLE_OF_BLOCKS, 1024>>>(input, splitter, indexA, indexB, len, BLOCK_SIZE);
                    CUDA_CHECK_ERROR
                    DPRINT_ARR_CUDA(splitter, SPLITTER_NUMBER)
                    DPRINT_ARR_CUDA(indexA, SPLITTER_NUMBER)
                    DPRINT_ARR_CUDA(indexB, SPLITTER_NUMBER)

                    // 2. ordina splitters
                    T * splitter_out = utils::cuda::allocate<T>(SPLITTER_NUMBER);
                    int * indexA_out   = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                    int * indexB_out   = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                    merge3_step<T, int, int>(splitter, splitter_out, 
                        indexA, indexA_out, 
                        indexB, indexB_out, 
                        SPLITTER_NUMBER, SPLITTER_PER_BLOCKS);
                    DPRINT_MSG("Splitter per block: %2d", SPLITTER_PER_BLOCKS)
                    DPRINT_ARR_CUDA(splitter_out, SPLITTER_NUMBER)
                    DPRINT_ARR_CUDA(indexA_out, SPLITTER_NUMBER)
                    DPRINT_ARR_CUDA(indexB_out, SPLITTER_NUMBER)

                    // 3. sistema l'indice finale di ogni blocco di splitter - l'ultimo contiene la lunghezza del blocco
                    fix_indexes_kernel<<<COUPLE_OF_BLOCKS, 1>>>(indexA_out, indexB_out, len, BLOCK_SIZE, SPLITTER_NUMBER, SPLITTER_PER_BLOCKS);
                    CUDA_CHECK_ERROR
                    DPRINT_ARR_CUDA(indexA_out, SPLITTER_NUMBER)
                    DPRINT_ARR_CUDA(indexB_out, SPLITTER_NUMBER)

                    // 4. eseguo il merge di porzioni di blocchi di dimensione uniforme
                    uniform_merge3_kernel<T, T2, T3><<<SPLITTER_NUMBER, 1>>>(input, output, indexA_out, indexB_out, val1_input, val1_output, val2_input, val2_output, len, BLOCK_SIZE);
                    CUDA_CHECK_ERROR

                    utils::cuda::deallocate(splitter);
                    utils::cuda::deallocate(indexA);
                    utils::cuda::deallocate(indexB);
                    utils::cuda::deallocate(splitter_out);
                    utils::cuda::deallocate(indexA_out);
                    utils::cuda::deallocate(indexB_out);
                }
                
                // 5. eventualmente copio il risultato dell' ultimo blocco di array rimasto spaiato
                if(BLOCK_NUMBER % 2 == 1) {
                    const int LAST_BLOCK_START = 2 * COUPLE_OF_BLOCKS * BLOCK_SIZE;
                    utils::cuda::copy<T>(      output + LAST_BLOCK_START,      input + LAST_BLOCK_START, len - LAST_BLOCK_START);
                    utils::cuda::copy<T2>(val1_output + LAST_BLOCK_START, val1_input + LAST_BLOCK_START, len - LAST_BLOCK_START);
                    utils::cuda::copy<T3>(val2_output + LAST_BLOCK_START, val2_input + LAST_BLOCK_START, len - LAST_BLOCK_START);
                }
            }
        }

        NUMERIC_TEMPLATE(T)
        inline void merge_step(T INPUT_ARRAY input, T * output, int len, int BLOCK_SIZE) {

            const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);

            if(BLOCK_SIZE <= MERGE_SMALL_MAX_BLOCK_SIZE) {
                DPRINT_MSG("Block size is %d: calling MERGE SMALL", BLOCK_SIZE)
                merge_small_step_kernel<T><<<DIV_THEN_CEIL(BLOCK_NUMBER, 2), MERGE_SMALL_MAX_BLOCK_SIZE>>>(input, output, len, BLOCK_SIZE);
            
            } else {
                DPRINT_MSG("Block size is %d: calling MERGE BIG", BLOCK_SIZE)
                const int COUPLE_OF_BLOCKS = DIV_THEN_CEIL(len, BLOCK_SIZE) / 2;
                const int SPLITTER_PER_BLOCKS = DIV_THEN_CEIL(BLOCK_SIZE, MERGE_BIG_SPLITTER_DISTANCE);
                const int SPLITTER_NUMBER = calculate_splitter_number(len, BLOCK_SIZE);

                // 1. genera splitter
                T * splitter = utils::cuda::allocate<T>(SPLITTER_NUMBER);
                int * indexA = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                int * indexB = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                generate_splitter_kernel<T><<<COUPLE_OF_BLOCKS, 1024>>>(input, splitter, indexA, indexB, len, BLOCK_SIZE);
                CUDA_CHECK_ERROR
                DPRINT_ARR_CUDA(splitter, SPLITTER_NUMBER)
                DPRINT_ARR_CUDA(indexA, SPLITTER_NUMBER)
                DPRINT_ARR_CUDA(indexB, SPLITTER_NUMBER)

                // 2. ordina splitters
                T * splitter_out = utils::cuda::allocate<T>(SPLITTER_NUMBER);
                int * indexA_out = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                int * indexB_out = utils::cuda::allocate<int>(SPLITTER_NUMBER);
                merge3_step<T, int, int>(
                    splitter, splitter_out, 
                    indexA, indexA_out, 
                    indexB, indexB_out, 
                    SPLITTER_NUMBER, SPLITTER_PER_BLOCKS);
                DPRINT_MSG("Splitter per block: %2d", SPLITTER_PER_BLOCKS)
                DPRINT_ARR_CUDA(splitter_out, SPLITTER_NUMBER)
                DPRINT_ARR_CUDA(indexA_out, SPLITTER_NUMBER)
                DPRINT_ARR_CUDA(indexB_out, SPLITTER_NUMBER)

                // 3. sistema l'indice finale di ogni blocco di splitter - l'ultimo contiene la lunghezza del blocco
                fix_indexes_kernel<<<COUPLE_OF_BLOCKS, 1>>>(indexA_out, indexB_out, len, BLOCK_SIZE, SPLITTER_NUMBER, SPLITTER_PER_BLOCKS);
                CUDA_CHECK_ERROR
                DPRINT_MSG("After fixing %d-element of couple...", 2*SPLITTER_PER_BLOCKS)
                DPRINT_ARR_CUDA(indexA_out, SPLITTER_NUMBER)
                DPRINT_ARR_CUDA(indexB_out, SPLITTER_NUMBER)

                // 4. eseguo il merge di porzioni di blocchi di dimensione uniforme
                uniform_merge_kernel<T><<<SPLITTER_NUMBER, 1>>>(input, output, indexA_out, indexB_out, len, BLOCK_SIZE);
                CUDA_CHECK_ERROR

                // 5. eventualmente copio il risultato dell' ultimo blocco di array rimasto spaiato
                if(BLOCK_NUMBER % 2 == 1) {
                    const int LAST_BLOCK_START = 2 * COUPLE_OF_BLOCKS * BLOCK_SIZE;
                    utils::cuda::copy<T>(output + LAST_BLOCK_START, input + LAST_BLOCK_START, len - LAST_BLOCK_START);
                }

                utils::cuda::deallocate(splitter);
                utils::cuda::deallocate(indexA);
                utils::cuda::deallocate(indexB);
                utils::cuda::deallocate(splitter_out);
                utils::cuda::deallocate(indexA_out);
                utils::cuda::deallocate(indexB_out);
            }
        }
    }

    namespace reference {

        NUMERIC_TEMPLATE3(T, T2, T3)
        inline void merge3_step(T INPUT_ARRAY input, T * output, T2 INPUT_ARRAY a_in, T2 * a_out, T3 INPUT_ARRAY b_in, T3 * b_out, int len, int BLOCK_SIZE) {
            
            bool all_three = a_in != NULL && a_out != NULL && b_in != NULL && b_out != NULL;
            const int BLOCK_NUMBER = DIV_THEN_CEIL(len, BLOCK_SIZE);

            for(int couple_block_id = 0; couple_block_id < DIV_THEN_CEIL(BLOCK_NUMBER, 2); couple_block_id++) {

                DPRINT_MSG("Processing couple_block_id %d\n", couple_block_id)

                int start_1 = 2 * couple_block_id * BLOCK_SIZE;
                int start_2 = min((2 * couple_block_id + 1) * BLOCK_SIZE, len);
                int end_1 = min((2 * couple_block_id + 1) * BLOCK_SIZE, len);
                int end_2 = min((2 * couple_block_id + 2) * BLOCK_SIZE, len);

                DPRINT_MSG("A[%d:%d] B[%d:%d]\n", start_1, end_1, start_2, end_2)

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
                utils::copy_array<T>(output + current_output, input + current_1, end_1 - current_1);
                if(all_three) utils::copy_array<T2>(a_out + current_output, a_in + current_1, end_1 - current_1);
                if(all_three) utils::copy_array<T3>(b_out + current_output, b_in + current_1, end_1 - current_1);

                // finisco le rimanenze del secondo blocco
                utils::copy_array<T>(output + current_output, input + current_2, end_2 - current_2);
                if(all_three) utils::copy_array<T2>(a_out + current_output, a_in + current_2, end_2 - current_2);
                if(all_three) utils::copy_array<T3>(b_out + current_output, b_in + current_2, end_2 - current_2);
            }
        }

        NUMERIC_TEMPLATE(T)
        inline void merge_step(T INPUT_ARRAY input, T * output, int len, int BLOCK_SIZE) {
            int * _ = NULL;
            merge3_step<T, int, int>(input, output, _, _, _, _, len, BLOCK_SIZE); 
        }
    
    }
}


#endif