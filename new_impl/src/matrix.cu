#include "matrix.hh"

namespace matrix {

    SparseMatrix::SparseMatrix(const int m, const int n, const int nnz, const MatrixInitialization mi) 
        : m(m), n(n), nnz(nnz) { 

        long long max_nnz = ((long long) m) *((long long) n);
        
        if(m <= 0 ) {
            throw std::invalid_argument("SparseMatrix m negative");
        }
        else if(n <= 0 ){
                throw std::invalid_argument("SparseMatrix n negative");
            }
        else if(nnz <= 0){
                throw std::invalid_argument("SparseMatrix nnz negative");
            }
        else if( nnz > max_nnz){
            throw std::invalid_argument("SparseMatrix multiplication");
            }

        this->csrRowPtr = new int[this->m+1]();
        //DPRINT_MSG("Allocating csrRowPtr: %p", this->csrRowPtr)
        this->csrColIdx = new int[this->nnz]();
        //DPRINT_MSG("Allocating csrColIdx: %p", this->csrColIdx)
        this->csrVal = new float[this->nnz]();
        //DPRINT_MSG("Allocating csrVal: %p", this->csrVal)

        if(mi == RANDOM_INITIALIZATION) {

            // 0. generate indices
            std::set<unsigned long long> indices_compact; // set prevents duplicate insertion
            while(indices_compact.size() < nnz) {
                int a = utils::random::generate(0, m-1);
                int b = utils::random::generate(0, n-1);
                unsigned long long x = (((long long)a) << 32) | ((long long)b);
                indices_compact.insert(x);
                //std::cout << indices_compact.size() << std::endl << std::flush;
            }

            // 2. fill values
            int i = 0;
            for(const unsigned long long& index : indices_compact) {

                int row = (int) (index >> 32);
                int col = (int) (index & 0x00000000FFFFFFFF);
                int val = utils::random::generate(1, 100);

                if(row < 0 || row > m) {
                    std::cout << "Error row\n" << std::flush;
                } else if(col < 0 || col > n) {
                    std::cout << "Error col\n" << std::flush;
                }

                csrRowPtr[row+1]++;
                csrColIdx[i] = col;
                csrVal[i] = (float) val;
                i++;
            }

            // 3. prefix_sum on csrRowPtr
            utils::prefix_sum(csrRowPtr, m+1);
        }
    }

    SparseMatrix::SparseMatrix(std::ifstream& mtx_file) {
       
        // ignoro l'header e i commenti
        while(mtx_file.peek() == '%') {
            mtx_file.ignore(2048, '\n');
        }

        // dimensioni e specifiche della matrice
        mtx_file >> m >> n >> nnz;
        // check dimensioni
        long long max_nnz = ((long long) m) *((long long) n);
        
        if(m <= 0 ) {
            throw std::invalid_argument("SparseMatrix MTX m negative");
        }
        else if(n <= 0 ){
                throw std::invalid_argument("SparseMatrix MTX n negative");
            }
        else if(nnz <= 0){
                throw std::invalid_argument("SparseMatrix MTX nnz negative");
                }
        else if( nnz > max_nnz){
            throw std::invalid_argument("SparseMatrix MTX multiplication");
        }

        // alloco lo spazio necessario
        int * inter;
        int * csrRowIdx     = new int[nnz]();
        int * csrRowPtrTemp = new int[m+1]();
        csrRowPtr           = new int[m+1]();
        csrColIdx           = new int[nnz]();
        csrVal              = new float[nnz]();

        // leggo da file
        double data;
        for(int i = 0; i < nnz; i++) {
            mtx_file >> csrRowIdx[i] >> csrColIdx[i] >> data;
            csrVal[i] = data;
        }
        // sistemo array puntatori
        procedures::reference::indexes_to_pointers(csrRowIdx, nnz, &inter, csrRowPtrTemp, m);
        procedures::reference::scan(csrRowPtrTemp, csrRowPtr, m+1);

        delete[] csrRowIdx;
        delete[] inter;
        delete[] csrRowPtrTemp;
    }

    SparseMatrix::~SparseMatrix() {
        delete[] csrRowPtr;
        delete[] csrColIdx;
        delete[] csrVal;
    }

    bool SparseMatrix::equals(SparseMatrix* sm) {
        return m == sm->m && n == sm->n && nnz == sm->nnz 
            && utils::equals(csrRowPtr, sm->csrRowPtr, m+1)
            && utils::equals(csrColIdx, sm->csrColIdx, nnz)
            && utils::equals(csrVal, sm->csrVal, nnz);
    }


    SparseMatrix* SparseMatrix::transpose(TranspositionMethod tm) {

        matrix::SparseMatrix *sm = this;

        int N = sm->n, M = sm->m, NNZ = sm->nnz;
        
        matrix::SparseMatrix *sm_out = new matrix::SparseMatrix(N, M, NNZ, matrix::ALL_ZEROS_INITIALIZATION);
        
        switch(tm) {
            case SERIAL:
                transposers::serial_csr2csc(
                    M, N, NNZ,
                    sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
                    sm_out->csrRowPtr, sm_out->csrColIdx, sm_out->csrVal
                ); break;
            case SCANTRANS:
                transposers::cuda_wrapper(
                    M, N, NNZ,
                    sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
                    sm_out->csrRowPtr, sm_out->csrColIdx, sm_out->csrVal,
                    transposers::scan_csr2csc
                ); break;
            case MERGETRANS:
                transposers::cuda_wrapper(
                    M, N, NNZ,
                    sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
                    sm_out->csrRowPtr, sm_out->csrColIdx, sm_out->csrVal,
                    transposers::merge_csr2csc
                ); break;
            case CUSPARSE1:
                transposers::cuda_wrapper(
                    M, N, NNZ,
                    sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
                    sm_out->csrRowPtr, sm_out->csrColIdx, sm_out->csrVal,
                    transposers::cusparse1_csr2csc
                ); break;
            case CUSPARSE2:
                transposers::cuda_wrapper(
                    M, N, NNZ,
                    sm->csrRowPtr, sm->csrColIdx, sm->csrVal,
                    sm_out->csrRowPtr, sm_out->csrColIdx, sm_out->csrVal,
                    transposers::cusparse2_csr2csc
                ); break;
            default:
                throw std::invalid_argument("unknown transposition method");
                break;
        }

        return sm_out;
    }

    void FullMatrix::fill_with_rand_numbers() {
        // creo la matrice "classica" e riempio i primi nnz elementi
        for (int i = 0; i < nnz; i++)
            matrix[i] = (float) utils::random::generate(1, 100);

        // mischio gli elementi nell'array
        std::shuffle(matrix, matrix+(n*m), utils::random::generator());
    }

    FullMatrix::FullMatrix(const int m, const int n, const int nnz, const MatrixInitialization mi) 
        : m(m), n(n), nnz(nnz)
    { 
        long long max_nnz = ((long long) m) *((long long) n);
        if(m <= 0 ) {
            throw std::invalid_argument("FullMatrix m negative");
        }
        else if(n <= 0 ){
                throw std::invalid_argument("FullMatrix n negative");
            }
        else if(nnz <= 0){
                throw std::invalid_argument("FullMatrix nnz negative");
                }
        else if( nnz > max_nnz){
            throw std::invalid_argument("FullMatrix multiplication");
        }
        matrix = new float[m*n]();

        if(mi == RANDOM_INITIALIZATION) {
            fill_with_rand_numbers();
        }
    }

    FullMatrix::FullMatrix(const SparseMatrix* sm) {

        m = sm->m;
        n = sm->n;
        nnz = sm->nnz;
        matrix = new float[n*m]();

        // genero indici di riga, in questo modo la coppia 
        // (row_indices, indices) mi porta ad avere la notazione COO
        int* csrRowIdx = new int[nnz];
        procedures::reference::pointers_to_indexes(sm->csrRowPtr, sm->m, csrRowIdx, nnz);

        // sistemo gli elementi nella matrice
        for(int i = 0; i < nnz; i++) {
            // estrai la colonna
            int col = sm->csrColIdx[i];
            // estrai la riga
            int row = csrRowIdx[i];
            // salvo il valore nella matrice di output
            matrix[row*n + col] = sm->csrVal[i];
        }

        delete[] csrRowIdx;
    }

    FullMatrix::~FullMatrix() {
        delete[] matrix;
    }

    bool FullMatrix::equals(FullMatrix* fm) {
        return m == fm->m && n == fm->n && nnz == fm->nnz 
            && utils::equals(matrix, fm->matrix, m*n);
    }

    FullMatrix* FullMatrix::transpose() {

        FullMatrix * fm = new FullMatrix(n, m, nnz, ALL_ZEROS_INITIALIZATION);

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                fm->matrix[j*m+i] = this->matrix[i*n+j];
            }
        }
        return fm;
    }

    SparseMatrix* FullMatrix::to_sparse() {

        SparseMatrix* sm = new SparseMatrix(m, n, nnz, ALL_ZEROS_INITIALIZATION);

        // 1. riempio i campi
        int i = 0;
        for(int row = 0; row < m; row++) {

            // devo aggiungere il primo elemento di ogni riga
            bool primoElementoRigaTrovato = false;

            for(int col = 0; col < n; col++) {
                
                // prendo il dato
                float cell = matrix[row * n + col];

                // se il dato non è nullo lo aggiungo
                if(cell != 0) {
                    sm->csrVal[i] = cell;
                    sm->csrColIdx[i] = col;

                    // se è il primo elemento, lo salvo nei puntatori agli indici
                    if( !primoElementoRigaTrovato ) {
                        sm->csrRowPtr[row] = i;
                        primoElementoRigaTrovato = true;
                    }

                    // incremento puntatore alla prossima cella di `data`
                    i++;
                }
            }

            // se non ho trovato il primo elemento della riga, allora metto 
            // il valore segnaposto -1 che viene poi sistemato successivamente
            if( !primoElementoRigaTrovato ) {
                sm->csrRowPtr[row] = -1;
            }
        }

        // 2. sistemo l'ultimo elemento di `indptr` 
        sm->csrRowPtr[m] = nnz;
        
        // 3. sistemo l'anomalia dei -1 nell'array indptr
        for(int i = m; i > 0; i--) {
            if(sm->csrRowPtr[i-1] == -1) {
                sm->csrRowPtr[i-1] = sm->csrRowPtr[i];
            }
        }

        return sm;
    }

} 