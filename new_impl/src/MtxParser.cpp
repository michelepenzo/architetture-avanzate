//#include "MtxParser.h"
//#include "procedures.hh"

#include <iostream>
#include <fstream>
#include <algorithm>
//#include <vector>
#include <math.h>


using namespace std;

int main(int argc, char* argv[]){
  	
  	int m, n, nnz;					
	int row, col;					
	float data;  
    int lenColIdx = 0, lenRowIdx = 0, last_row = 1;;

    std::string file_name = "/home/michele/Downloads/new/test1.mtx";

    ifstream fin(file_name);        // carico il file
    while(fin.peek() == '%')        // ignoro l'header e i commenti
        fin.ignore(2048, '\n');

    fin >> m >> n >> nnz;           // leggo la prima riga
	
	// alloco lo spazio per i tre vettori
	int *csrVal    = new int[nnz];
  	int *csrRowIdx = new int[nnz];    
  	int *csrColIdx = new int[m+1];

    csrRowIdx[0] = 0; 	
    lenRowIdx += 1;
    
    for (int i = 0; i < nnz; i++)   // ciclo su tutte le altre righe
    {
        
        fin >> row >> col >> data;  // leggo la riga

        csrColIdx[i] = col-1;
        csrVal[i] = data;
        lenColIdx += 1;

        if(row > last_row)
        {
            last_row = row;
            csrRowIdx[lenRowIdx] = lenColIdx-1;
            lenRowIdx += 1;
        }
    }
    csrRowIdx[lenRowIdx] = lenColIdx;
    fin.close();

    // =====================================================================
    // cout<< "csrVal= ";
    // for (int i = 0; i < nnz; ++i)
    //    	std::cout << csrVal[i] << " ";
    
    // cout<< "\ncsrColIdx= ";
    // for (int i = 0; i < nnz; ++i)
    //    	std::cout << csrColIdx[i] << " ";
    
    // cout<< "\ncsrRowIdx= ";
    // for (int i = 0; i < m+1; ++i)
    //    	std::cout << csrRowIdx[i] << " ";
    
    // =====================================================================

    int *csrVal_out    = new int[nnz];
    int *csrRowIdx_out = new int[nnz];    
    int *csrColIdx = new int[m+1];

    procedures::cuda::sort3(csrRowIdx, csrRowIdx_out, lenRowIdx, csrColIdx, csrColIdx_out, csrVal, csrVal_out);

    procedures::cuda::index_to_pointers();


  	return 0;
} 
