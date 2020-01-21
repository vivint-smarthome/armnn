#!/bin/bash

set -e

thisDir=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
NPROC=4

cd $thisDir
# Download code here
git clone https://github.com/vivint-smarthome/ComputeLibrary.git &
curl -LO https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2 && \
tar xf boost_1_64_0.tar.bz2 && \
rm -rf boost_1_64_0.tar.bz2 && \
cd boost_1_64_0 && \
echo "using gcc : arm : aarch64-linux-gnu-g++ ;" > user_config.jam &
curl -Lo protobuf-3.5.1.tar.gz https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz && \
  tar -xzf protobuf-3.5.1.tar.gz && \
  rm -rf protobuf-3.5.1.tar.gz &
git clone https://github.com/tensorflow/tensorflow.git && \
  cd tensorflow/ && \
  git checkout a0043f9262dc1b0e7dc4bdf3a7f0ef0bebc4891e &
git clone https://github.com/google/flatbuffers.git && \
  mkdir flatbuffers_x86_64 && \
  cp -a flatbuffers/* flatbuffers_x86_64 &

wait

rm -rf $thisDir/build && mkdir -p $thisDir/build

# Build compute library
cd $thisDir/ComputeLibrary
if ! which scons >/dev/null 2>&1 ; then
  sudo dnf install scons-python3 -y
fi
scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" -j${NPROC} internal_only=0 benchmark_tests=0 validation_tests=0 Werror=0

# Build boost
cd $thisDir/boost_1_64_0
mkdir -p $thisDir/build/boost_arm64_install
./bootstrap.sh --prefix=$thisDir/build/boost_arm64_install
./b2 install toolset=gcc-arm link=static cxxflags=-fPIC --with-filesystem --with-test --with-log --with-program_options -j${NPROC} --user-config=user_config.jam --prefix=$thisDir/build/boost_arm64_install

# Build protobuf
cd $thisDir/protobuf-3.5.1
mkdir -p $thisDir/build/protobuf/x86_64
cd $thisDir/build/protobuf/x86_64
$thisDir/protobuf-3.5.1/configure --prefix=$thisDir/build/protobuf/x86_64
make install -j${NPROC}

mkdir -p $thisDir/build/protobuf/aarch64
cd $thisDir/build/protobuf/aarch64
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ $thisDir/protobuf-3.5.1/configure --host=aarch64-linux --prefix=$thisDir/build/arm64_pb_install --with-protoc=$(which protoc)
make -j${NPROC}

# Build tensorflow protobufs
cd $thisDir/tensorflow
$thisDir/scripts/generate_tensorflow_protobuf.sh $thisDir/build/tensorflow-protobuf $thisDir/build/protobuf/x86_64

# Build flatbuffers
cd $thisDir/flatbuffers
CXX=aarch64-linux-gnu-g++ \
CC=aarch64-linux-gnu-gcc \
cmake -G "Unix Makefiles" \
-DCMAKE_BUILD_TYPE=Release \
-DFLATBUFFERS_BUILD_TESTS=OFF \
-DFLATBUFFERS_BUILD_FLATC=ON \
-DCMAKE_CXX_FLAGS=-fPIC
make

cd $thisDir/flatbuffers_x86_64
cmake -G "Unix Makefiles" \
-DCMAKE_BUILD_TYPE=Release \
-DFLATBUFFERS_BUILD_TESTS=OFF \
-DFLATBUFFERS_BUILD_FLATC=ON \
-DCMAKE_CXX_FLAGS=-fPIC
make

# Build ArmNN
#cd $thisDir/armnn
#rm -rf build
#mkdir build
#cd build

rm -rf $thisDir/armnn-build
mkdir $thisDir/armnn-build
cd $thisDir/armnn-build

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
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address  -fsanitize=leak -g" \
  -DCMAKE_C_FLAGS="-fsanitize=address  -fsanitize=leak -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address  -fsanitize=leak" \
  -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=address  -fsanitize=leak"

make -j$((NPROC * 4))

