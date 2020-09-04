#!/bin/bash

make clean

s=$( cat /usr/local/cuda/version.txt )
n="$(cut -d' ' -f3 <<<"$s")"
version="$(cut -d'.' -f1 <<<"$n")"
export NVCC_VERSION="$version"

make 
make test