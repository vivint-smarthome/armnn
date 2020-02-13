//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "armnn/ArmNN.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/// A simple example of using the ArmNN SDK API. In this sample, the users single input number is multiplied by 1.0f
/// using a fully connected layer with a single neuron to produce an output number that is the same as the input.
int main(int argc, char **argv)
{
    using namespace armnn;
    using namespace armnnTfLiteParser;

    if (argc != 3) {
        std::cout << "Usage: CompSample model.tflite image.rgb" << std::endl;
        return 1;
    }

    std::string model(argv[1]);
    std::string image(argv[2]);

    //std::cout << "Starting" << std::endl;

    //std::cout << "Reading image data from disk" << std::endl;
    std::vector<uint8_t> inputData;
    std::array<uint8_t, 4096> buff;
    int fd = open(image.c_str(), O_RDONLY);
    int sz = 0;
    while ((sz = read(fd, buff.data(), 4096)) > 0) {
        inputData.insert(inputData.end(), buff.begin(), buff.begin() + sz);
    }
    close(fd);


    // Construct ArmNN network
    armnn::NetworkId networkIdentifier;
    ITfLiteParserPtr parser = ITfLiteParser::Create();
    //std::cout << "Loading network from binary file" << std::endl;
    INetworkPtr tfNetwork = parser->CreateNetworkFromBinaryFile(model.c_str());

    // NOTE: ORIG for dbcp tflite
    BindingPointInfo input_binding = parser->GetNetworkInputBindingInfo(0, "normalized_input_image_tensor");
    // NOTE: for mobilenet
    // BindingPointInfo input_binding = parser->GetNetworkInputBindingInfo(0, "input");
    int input_binding_id = std::get<0>(input_binding);
    TensorInfo input_tensor_info = std::get<1>(input_binding);
    //std::cout << "Binding id: " << input_binding_id << std::endl;

    BindingPointInfo bbox_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess");
    BindingPointInfo class_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:1");
    BindingPointInfo score_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:2");
    BindingPointInfo num_detection_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:3");

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    //std::cout << "Creating runtime" << std::endl;
    IRuntimePtr run = IRuntime::Create(options);

    // Optimise ArmNN network
    //std::cout << "Optimizing network" << std::endl;
    armnn::IOptimizedNetworkPtr optNet = Optimize(*tfNetwork, {Compute::CpuAcc, Compute::CpuRef}, run->GetDeviceSpec());
    optNet->PrintGraph();

    // Load graph into runtime
    //std::cout << "Loading network" << std::endl;
    std::string error;
    run->LoadNetwork(networkIdentifier, std::move(optNet), error);
    if (error != "") {
        std::cout << "Error: " << error << std::endl;
    }

    //Creates structures for inputs and outputs.
    std::vector<float> bboxes(80);
    std::vector<float> classes(20);
    std::vector<float> scores(20);
    std::vector<float> num_detections(1);

    std::fill(bboxes.begin(), bboxes.end(), 0.0f);
    std::fill(classes.begin(), classes.end(), 0.0f);
    std::fill(scores.begin(), scores.end(), 0.0f);
    std::fill(num_detections.begin(), num_detections.end(), 0.0f);

    //std::cout << "Building const tensor for input" << std::endl;
    size_t in_tensor_size = std::min((size_t)input_tensor_info.GetNumBytes(), inputData.size());

    inputData.resize(in_tensor_size);

    ConstTensor input_tensor = armnn::ConstTensor(input_tensor_info, inputData);

    //std::cout << "Getting input tensors" << std::endl;
    armnn::InputTensors inputTensors{
        {input_binding_id, input_tensor}
//      {0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, input_binding_id), inputData.data())}
    };

    //std::cout << "Getting output tensors" << std::endl;
    armnn::OutputTensors outputTensors{
      {std::get<0>(bbox_binding), armnn::Tensor(std::get<1>(bbox_binding), bboxes.data())}, 
      {std::get<0>(class_binding), armnn::Tensor(std::get<1>(class_binding), classes.data())},
      {std::get<0>(score_binding), armnn::Tensor(std::get<1>(score_binding), scores.data())},
      {std::get<0>(num_detection_binding), armnn::Tensor(std::get<1>(num_detection_binding), num_detections.data())},
    };

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Output results
    size_t max_bbox = std::min(bboxes.size(), (size_t)num_detections[0]*4);
    size_t max_class = std::min(classes.size(), (size_t)num_detections[0]);
    size_t max_score = std::min(scores.size(), (size_t)num_detections[0]);

    for (int i = 0; i < max_bbox; i++) {
        printf("BBox %d: %.8g\n", i, std::min(std::max(bboxes[i], 0.0f), 1.0f));
    }
    for (int i = 0; i < max_class; i++) {
        printf("Class %d: %.8g\n", i, classes[i]);
    }
    for (int i = 0; i < max_score; i++) {
        printf("Score %d: %.8g\n", i, scores[i]);
    }
    for (int i = 0; i < num_detections.size(); i++) {
        printf("NumDetections %d: %g\n", i, num_detections[i]);
    }

    return 0;
}

