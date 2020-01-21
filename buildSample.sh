#!/bin/bash

set -e
cd $(dirname ${BASH_SOURCE[0]})
#for i in $(seq 0 183)
#do
#echo "*******************************************************"
#echo "Setting var to $i"
#sed "s/REPLACE/$i/" ../../samples/SimpleSample.cpp.replace > ../../samples/SimpleSample.cpp

rm -f SimpleSample

aarch64-unknown-linux-gnu-g++ -DARMCOMPUTENEON_ENABLED -DARMNNREF_ENABLED -DARMNN_SERIALIZER -DARMNN_SERIALIZER_SCHEMA_PATH="/armnn-tflite/armnn/src/armnnSerializer/ArmnnSchema.fbs" -DARMNN_TF_LITE_PARSER -DARMNN_TF_LITE_SCHEMA_PATH="/armnn-tflite/tensorflow/tensorflow/lite/schema/schema.fbs" -DBOOST_ALL_NO_LIB -I/armnn-tflite/armnn/include -isystem /armnn-tflite/build/boost_arm64_install/include -isystem /armnn-tflite/ComputeLibrary -isystem /armnn-tflite/ComputeLibrary/include -isystem /armnn-tflite/armnn/third-party -std=c++14 -Wall -Werror -Wold-style-cast -Wno-missing-braces -Wconversion -Wsign-conversion -O2 -DNDEBUG -DNDEBUG -O3 -fPIE -o CMakeFiles/SimpleSample.dir/SimpleSample.cpp.o -c /armnn-tflite/armnn/samples/SimpleSample.cpp

aarch64-unknown-linux-gnu-g++ -std=c++14 -Wall -Werror -Wold-style-cast -Wno-missing-braces -Wconversion -Wsign-conversion -O2 -DNDEBUG -DNDEBUG -O3 -rdynamic CMakeFiles/SimpleSample.dir/SimpleSample.cpp.o -o SimpleSample -L/armnn-tflite/build/boost_arm64_install/lib -Wl,-rpath,/armnn-tflite/build/boost_arm64_install/lib:/armnn-tflite/armnn/build ../libarmnn.so.19.11 ../libarmnnTfLiteParser.so.19.11 -lpthread ../libarmnnUtils.a -ldl /armnn-tflite/build/boost_arm64_install/lib/libboost_log.a /armnn-tflite/build/boost_arm64_install/lib/libboost_thread.a /armnn-tflite/build/boost_arm64_install/lib/libboost_system.a /armnn-tflite/build/boost_arm64_install/lib/libboost_filesystem.a /armnn-tflite/ComputeLibrary/build/libarm_compute_core-static.a /armnn-tflite/ComputeLibrary/build/libarm_compute-static.a /armnn-tflite/ComputeLibrary/build/libarm_compute_core-static.a /armnn-tflite/ComputeLibrary/build/libarm_compute-static.a

scp SimpleSample 192.168.5.13:/mnt/analytics/tmp/samples/

ssh 192.168.5.13 LD_LIBRARY_PATH=/mnt/analytics/tmp /mnt/analytics/tmp/samples/SimpleSample
#done
