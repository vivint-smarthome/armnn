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

/// A simple example of using the ArmNN SDK API. In this sample, the users single input number is multiplied by 1.0f
/// using a fully connected layer with a single neuron to produce an output number that is the same as the input.
int main()
{
    using namespace armnn;
    using namespace armnnTfLiteParser;

    std::cout << "Starting" << std::endl;

    // Construct ArmNN network
    armnn::NetworkId networkIdentifier;
    ITfLiteParserPtr parser = ITfLiteParser::Create();
    std::cout << "Loading network from binary file" << std::endl;
    INetworkPtr tfNetwork = parser->CreateNetworkFromBinaryFile("./active.tflite");

    // NOTE: ORIG for dbcp tflite
    BindingPointInfo input_binding = parser->GetNetworkInputBindingInfo(0, "normalized_input_image_tensor");
    // NOTE: for mobilenet
    // BindingPointInfo input_binding = parser->GetNetworkInputBindingInfo(0, "input");
    int input_binding_id = std::get<0>(input_binding);
    TensorInfo input_tensor_info = std::get<1>(input_binding);
    std::cout << "Binding id: " << input_binding_id << std::endl;

    BindingPointInfo bbox_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess");
    BindingPointInfo class_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:1");
    BindingPointInfo score_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:2");
    BindingPointInfo num_detection_binding = parser->GetNetworkOutputBindingInfo(0, "TFLite_Detection_PostProcess:3");

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    std::cout << "Creating runtime" << std::endl;
    IRuntimePtr run = IRuntime::Create(options);

    // Optimise ArmNN network
    std::cout << "Optimizing network" << std::endl;
    armnn::IOptimizedNetworkPtr optNet = Optimize(*tfNetwork, {Compute::CpuAcc, Compute::CpuRef}, run->GetDeviceSpec());
    optNet->PrintGraph();

    // Load graph into runtime
    std::cout << "Loading network" << std::endl;
    std::string error;
    run->LoadNetwork(networkIdentifier, std::move(optNet), error);
    if (error != "") {
        std::cout << "Error: " << error << std::endl;
    }

    //Creates structures for inputs and outputs.
    std::vector<uint8_t> inputData(input_tensor_info.GetNumBytes());
    std::vector<float> bboxes(80);
    std::vector<float> classes(20);
    std::vector<float> scores(20);
    std::vector<float> num_detections(1);

    std::cout << "Building const tensor for input" << std::endl;
    ConstTensor input_tensor = armnn::ConstTensor(input_tensor_info, inputData);

    std::cout << "Getting input tensors" << std::endl;
    armnn::InputTensors inputTensors{
        {input_binding_id, input_tensor}
//      {0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, input_binding_id), inputData.data())}
    };

    std::cout << "Getting output tensors" << std::endl;
    armnn::OutputTensors outputTensors{
      {std::get<0>(bbox_binding), armnn::Tensor(std::get<1>(bbox_binding), bboxes.data())}, 
      {std::get<0>(class_binding), armnn::Tensor(std::get<1>(class_binding), classes.data())},
      {std::get<0>(score_binding), armnn::Tensor(std::get<1>(score_binding), scores.data())},
      {std::get<0>(num_detection_binding), armnn::Tensor(std::get<1>(num_detection_binding), num_detections.data())},
    };

    // Execute network
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int runs = 10;
    for (int i = 0; i < runs; i++) {
        std::cout << "Executing network" << std::endl;
        run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        std::cout << "Executed network" << std::endl;
    }
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::cout << "Milliseconds per run: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / runs << std::endl;
    return 0;

}

