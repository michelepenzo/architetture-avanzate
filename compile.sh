#!/bin/bash
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/final_project_transpose.cu -I"$DIR"/include -o final_project_transpose
