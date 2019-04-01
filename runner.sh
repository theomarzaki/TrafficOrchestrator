#!/bin/bash

set -eux

mkdir -p build
rm -rf build/*
cd build
cmake ..
make
./exchange
