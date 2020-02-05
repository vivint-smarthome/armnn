//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Stack.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void Stack(const StackQueueDescriptor& data,
        std::vector<std::unique_ptr<Decoder<float>>>& inputs,
        Encoder<float>& output)
{
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);
    const TensorInfo& inputInfo = GetTensorInfo(data.m_Inputs[0]);

    unsigned int outputNumDims = outputInfo.GetNumDimensions();
    unsigned int inputNumDims = inputInfo.GetNumDimensions();

    const armnn::TensorShape& outputDims = outputInfo.GetShape();
    const armnn::TensorShape& inputDims = inputInfo.GetShape();

    unsigned int axis = data.m_Parameters.m_Axis;

    // Initialise output data
    unsigned int numOutputElements = 1;
    for (unsigned int i=0; i<outputNumDims; ++i)
    {
        numOutputElements *= outputDims[i];
    }

    const unsigned int iNumTensors = static_cast<unsigned int>(data.m_Inputs.size());

    std::vector<unsigned int> osz;
    for (int i = 0; i < outputNumDims; i++) {
        osz.push_back(outputDims[i]);
    }

    std::vector<unsigned int> isz;
    isz.push_back(iNumTensors);
    for (int i = 0; i < inputNumDims; i++) {
        isz.push_back(inputDims[i]);
    }

    // the input coordinates
    // iCoordinates[0] -> the input tensor number
    // iCoordinates[n] -> dimension n iterator
    // ...
    // iCoordinates[iCoordinates.size()-2] -> the last real tensor
    // iCoordinates[iCoordinates.size()-1] -> always 0
    std::vector<unsigned int> iCoordinates;
    for (int i = 0; i < inputNumDims + 2; i++) {
        iCoordinates.push_back(0);
    }

    // Array of pointers used to map the output coordinates to the input ones, in accordance with the axis
    // This array is initialized with &iCoordinates[size-1] since this will be always zero
    std::vector<unsigned int *> oCoordinates;
    for (int i = 0; i < outputNumDims; i++) {
        oCoordinates.push_back(&iCoordinates[iCoordinates.size()]);
    }

    // Set the axis coordinate
    oCoordinates[axis] = &iCoordinates[0];

    // Map the output coordinates, accounting for the axis
    unsigned int dim_shift = 0;
    for(unsigned int dim = 0; dim < inputNumDims; ++dim)
    {
        if(dim == axis)
        {
            dim_shift++;
        }
        oCoordinates[dim + dim_shift] = &iCoordinates[dim + 1];
    }

    // Stack tensors

    size_t in_idx = 0;
    while (iCoordinates[0] < iNumTensors) {
        /*
        size_t out_idx2 = 
        (*oCoordinates[0])  * osz[5] * osz[4] * osz[3] * osz[2] * osz[1] +
               (*oCoordinates[1]) * osz[5] * osz[4] * osz[3] * osz[2] +
               (*oCoordinates[2])  * osz[5] * osz[4] * osz[3] +
               (*oCoordinates[3])  * osz[5] * osz[4] +
               (*oCoordinates[4])  * osz[5] + 
               (*oCoordinates[5]);

        size_t in_idx2 =
            iCoordinates[1] * isz[5] * isz[4] * isz[3] * isz[2] +
            iCoordinates[2] * isz[5] * isz[4] * isz[3] +
            iCoordinates[3] * isz[5] * isz[4] +
            iCoordinates[4] * isz[5] +
            iCoordinates[5];
        */

        size_t out_idx = 0;
        for (int i = 0; i < oCoordinates.size(); i++) {
            size_t row_tmp = (*oCoordinates[i]);
            for (int j = i+1; j < oCoordinates.size(); j++) {
                row_tmp *= osz[j];
            }
            out_idx += row_tmp;
        }

        /*
        size_t in_idx = 0;
        for (int i = 1; i < iCoordinates.size(); i++) {
            size_t row_tmp = iCoordinates[i];
            for (int j = i+1; j < iCoordinates.size()-1; j++) {
                row_tmp *= isz[j];
            }
            in_idx += row_tmp;
        }
        */

        // Write data
        output[out_idx];
        (*inputs[iCoordinates[0]])[in_idx];
        output.Set(inputs[iCoordinates[0]]->Get());

        // Update iCoordinates
        size_t old_tensor = iCoordinates[0];
        iCoordinates[iCoordinates.size()-2]++;
        for (ssize_t i = iCoordinates.size()-2; i >= 1; i--) {
            if (iCoordinates[i] >= isz[i]) {
                iCoordinates[i] = 0;
                iCoordinates[i-1]++;
            }
            else {
                break;
            }
        }

        in_idx++;
        if (old_tensor != iCoordinates[0]) {
            in_idx = 0;
        }
    }
}
} // namespace armnn
