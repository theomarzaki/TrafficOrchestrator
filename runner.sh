#!/bin/bash

set -eux

shopt -s extglob

mkdir -p build

cd build

rm -r !(libtorch) || rm -rf *  #prevents the need to re-download libtorch when recompiling again
#TODO remove the link in CMakeFile

cmake ..
make
./exchange
