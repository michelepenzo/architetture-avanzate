#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

class SparseMatrixCsr {

private:

    float* data;
    int* indices;
    int* indptr;
    int N;

    // funzione di supporto che ritorna il numero di elementi non-zero all'interno
    // della matrice N*N passata in input
    int count_nonzero(const float* matrix, const int N);

    // funzione di supporto che partendo dal CSR crea il vettore degli indici di riga,
    // in questo modo la tupla (data, row_indices, [col_]indices) rappresenta la stessa
    // matrice in formato COO
    void generate_row_indices(int* row_indices);

public:

    // costrutture vuoto
    SparseMatrixCsr() { } 

    // costruttore a partire da una matrice in input N*N
    SparseMatrixCsr(const float* matrix, const int N);

    // distruttore
    ~SparseMatrixCsr();

    // ritorna nel formato matrice originario
    void to_matrix(float* output_matrix);

    // traspone la matrice, che è equivalente a portarla nel formato CSC
    void transpose();

    // stampa in formato CSR
    void print();

    // stampa in formato COO
    void print_coo();

    // stampa originale N x N
    void print_original();

};

int SparseMatrixCsr::count_nonzero(const float* matrix, const int N) {
    int nonzeros = 0;
    for(int i = 0; i < N*N; i++) {
        if( matrix[i] != 0 ) {
            nonzeros++;
        }
    }
    return nonzeros;
}

void SparseMatrixCsr::generate_row_indices(int* row_indices) {

    for(int row = 0; row < N; row++) {

        int row_start = this->indptr[row];
        int row_end = this->indptr[row+1];

        for(int j = row_start; j < row_end; j++) {
            row_indices[j] = row;
        }
    }
}

SparseMatrixCsr::SparseMatrixCsr(const float* matrix, const int N)
{
    // guarda quanti elementi non nulli ci siano nella matrice
    int nonzeros = this->count_nonzero(matrix, N);
    // alloca della giusta dimensione gli array
    this->data    = new float[nonzeros];
    this->indices = new int[nonzeros];
    this->indptr  = new int[N+1]; // un elemento x riga + un elemento finale con la dimensione N della matrice
    this->N = N;
    // riempio i campi
    int i = 0;
    for(int row = 0; row < N; row++) {

        // devo aggiungere il primo elemento di ogni riga
        bool primoElementoRigaTrovato = false;

        for(int col = 0; col < N; col++) {
            
            // prendo il dato
            float cell = matrix[row * N + col];

            // se il dato non è nullo lo aggiungo
            if(cell != 0) {
                this->data[i] = cell;
                this->indices[i] = col;

                // se è il primo elemento, lo salvo nei puntatori agli indici
                if( !primoElementoRigaTrovato ) {
                    this->indptr[row] = i;
                    primoElementoRigaTrovato = true;
                }

                // incremento puntatore alla prossima cella di `data`
                i++;
            }
        }

        // se non ho trovato il primo elemento della riga, allora metto 
        // il valore segnaposto -1 che viene poi sistemato successivamente
        if( !primoElementoRigaTrovato ) {
            this->indptr[row] = -1;
        }
    }
    // l'ultimo elemento di `indptr` 
    this->indptr[N] = nonzeros;
    // sistemo l'anomalia dei -1 nell'array indptr
    for(int i = N; i > 0; i--) {
        if(this->indptr[i-1] == -1) {
            this->indptr[i-1] = this->indptr[i];
        }
    }
}

SparseMatrixCsr::~SparseMatrixCsr() {
    delete this->data;
    delete this->indices;
    delete this->indptr;
}

void SparseMatrixCsr::to_matrix(float* output_matrix) {

    int elements = this->indptr[this->N];

    // genero indici di riga, in questo modo la coppia (row_indices, indices)
    // mi porta ad avere la notazione COO
    int* row_indices = new int[elements];
    this->generate_row_indices(row_indices);

    // sistemo gli elementi nella matrice
    for(int i = 0; i < elements; i++) {

        // estrai l'elemento
        float element = this->data[i];
        // estrai la colonna
        int col = this->indices[i];
        // estrai la riga
        int row = row_indices[i];
        // salvo il valore nella matrice di output
        output_matrix[row*this->N + col] = element;
    }

    delete row_indices;
}

template <class T>
void swap(T& a, T& b){
    T tmp = a; a = b; b = tmp;
}

void SparseMatrixCsr::transpose() {

    int elements = this->indptr[this->N];

    // 1. genero gli indici di riga
    int* row_indices = new int[elements];
    this->generate_row_indices(row_indices);

    // 2. ordino per `indices`
    // O(n^2) -> vedi di usare std::sort incapsulando in una std::tuple gli elementi dei 4 vettori 
    // oppure definisci comparator che estrae indici tramite aritmetica dei puntatori
    for(int i = 0; i < elements-1; i++) {
        for(int j = i + 1; j < elements; j++) {
            if( this->indices[i] > this->indices[j] ) {

                swap(this->data[i], this->data[j]);
                swap(this->indices[i], this->indices[j]);
                swap(this->indptr[i], this->indptr[j]);
                swap(row_indices[i], row_indices[j]);
            }
        }
    }

    // 3. inverti le righe con le colonne: hai trasposto quindi portato da CSR a CSC
    int* col_indices = this->indices;
    this->indices = row_indices;

    // 4. compatta il vecchio `col_indices` per farlo diventare l'array dei puntatori alle colonne
    

    // 5. dealloca `col_indices` e ritorna
    delete col_indices;
}

void SparseMatrixCsr::print() {

    int elements = this->indptr[this->N];
    
    std::cout << "Stampa formato CSR" << std::endl;
    std::cout << "   Data: ";
    for(int i = 0; i < elements; i++) {
        std::cout << this->data[i] << " ";
    }
    std::cout << "\nIndices: ";
    for(int i = 0; i < elements; i++) {
        std::cout << this->indices[i] << " ";
    }
    std::cout << "\n Indptr: ";
    for(int i = 0; i < N+1; i++) {
        std::cout << this->indptr[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

void SparseMatrixCsr::print_coo() {
    
    int elements = this->indptr[this->N];

    int* row_indices = new int[elements];
    this->generate_row_indices(row_indices);

    std::cout << "Stampa formato COO" << std::endl;
    std::cout << "       Values: ";
    for(int i = 0; i < elements; i++) {
        std::cout << this->data[i] << " ";
    }
    std::cout << "\nColumns coord: ";
    for(int i = 0; i < elements; i++) {
        std::cout << this->indices[i] << " ";
    }
    std::cout << "\n   Rows coord: ";
    for(int i = 0; i < elements; i++) {
        std::cout << row_indices[i] << " ";
    }
    std::cout << std::endl << std::endl;

    delete row_indices;
}

void SparseMatrixCsr::print_original() {

    float* output_matrix = new float[N*N];
    this->to_matrix(output_matrix);

    std::cout << "Stampa formato N*N" << std::endl;
    for(int i = 0; i < this->N; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << output_matrix[i*this->N+j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl << std::endl;

    delete output_matrix;
}


int main() {

    std::cout << "=================== TEST #1 ===================\n";

    // 0 2 3
    // 0 5 0
    // 7 0 9

    // CSR data:    2 3 5 7 9
    // CSR indices: 1 2 1 0 2 # colonne
    // CSR indptr:  0 2 3 (5)
    // ricostruisco righe CRC
    // CSR righe:   0 0 1 2 2 

    // CSC data:    7 2 5 3 9
    // CSC indices: 2 0 1 0 2 # righe
    // CSC indptr:  0 1 3 (5)
    // ricostruisco colonne CSC
    // CSC colonne: 0 1 1 2 2 

    const int N = 3;
    float* matrix = new float[N*N] { 0, 2, 3, 0, 5, 0, 7, 0, 9 };

    SparseMatrixCsr sparse(matrix, N); // creazione matrice CSR
    sparse.print();                    // stampa della matrice CSR
    sparse.print_coo();                // stampa della matrice in formato COO
    sparse.print_original();           // stampa della matrice ri-creata

    std::cout << "=================== TEST #2 ===================\n";

    // matrix
    // 1 0 0 
    // 2 3 4
    // 0 0 0 

    // CSR
    // val 1 2 3 4 
    // col 0 0 1 2 
    // row 0 1 4 4 (0 1 * DIM)

    // CSC
    // val 1 2 3 4 
    // row 0 1 1 1 
    // col 0 2 3 4 (0 2 3 DIM)

    float* matrix2 = new float[N*N] { 1, 0, 0, 2, 3, 4, 0, 0, 0 };

    SparseMatrixCsr sparse2(matrix2, N); 
    sparse2.print();                    
    sparse2.print_coo();                
    sparse2.print_original();           

    delete matrix, matrix2; 
    return 0;
}


    

