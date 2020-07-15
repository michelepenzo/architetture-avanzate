#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/SparseTranpose.cu -I"$DIR"/include -o mat_sparse_transpose
