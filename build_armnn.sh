#!/bin/bash

set -e

thisDir=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
NPROC=4

# Build ArmNN
rm -rf armnn-build
mkdir armnn-build
cd armnn-build

CXX=aarch64-linux-gnu-g++ \
CC=aarch64-linux-gnu-gcc \
cmake .. \
  -DARMCOMPUTE_ROOT=$thisDir/ComputeLibrary \
  -DARMCOMPUTE_BUILD_DIR=$thisDir/ComputeLibrary/build \
  -DBOOST_ROOT=$thisDir/build/boost_arm64_install \
  -DTF_GENERATED_SOURCES=$thisDir/build/tensorflow-protobuf \
  -DPROTOBUF_ROOT=$thisDir/build/aarch64 \
  -DBUILD_TF_LITE_PARSER=1 \
  -DBUILD_ARMNN_SERIALIZER=1 \
  -DBUILD_ARMNN_QUANTIZER=1 \
  -DTF_LITE_GENERATED_PATH=$thisDir/tensorflow/tensorflow/lite/schema \
  -DFLATBUFFERS_ROOT=$thisDir/flatbuffers \
  -DFLATBUFFERS_LIBRARY=$thisDir/flatbuffers/libflatbuffers.a \
  -DFLATC=$thisDir/flatbuffers_x86_64/flatc \
  -DARMCOMPUTENEON=1 \
  -DARMNNREF=1 \
  -DBUILD_SAMPLE_APP=1 \
  -DCMAKE_BUILD_TYPE=Debug
#  -DCMAKE_CXX_FLAGS="-fsanitize=address  -fsanitize=leak -g" \
#  -DCMAKE_C_FLAGS="-fsanitize=address  -fsanitize=leak -g" \
#  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address  -fsanitize=leak" \
#  -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address  -fsanitize=leak"

make -j$((NPROC * 4))

