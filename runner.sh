#!/bin/bash

cd build

rm -rf *

cmake -DCMAKE_PREFIX_PATH=/communication_layer/libtorch ..

make

./exchange
