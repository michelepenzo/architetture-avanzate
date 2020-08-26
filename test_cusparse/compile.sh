#!/bin/bash
DIR=`dirname $0`
SRC_FILE="$DIR"/test_cusparse.cu
MID_FILE=test_cusparse.o
OUT_FILE=test_cusparse
PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2020/compilers/bin:$PATH

echo "First step..."
nvcc -c -std=c++11 -I/usr/local/cuda/include -I"$DIR"/include $SRC_FILE

echo "Second step..." # -lcublas -lcusolver are not needed to compile right now
g++ -fopenmp -o $OUT_FILE $MID_FILE -L/usr/local/cuda/lib64 -lcudart -lcusparse

echo "Done!"