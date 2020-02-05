//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

// TODO

#include "RefReduceMaxWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Profiling.hpp"

#include <cstring>

namespace armnn
{

void RefReduceMaxWorkload::Execute() const
{
    throw Exception("RefReduceMaxWorkload::Execute not implemented yet");
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefReduceMaxWorkload_Execute");

    void* output = GetOutputTensorData<void>(0, m_Data);
    const void* input = GetInputTensorData<void>(0, m_Data);
    unsigned int numBytes = GetTensorInfo(m_Data.m_Inputs[0]).GetNumBytes();
    memcpy(output, input, numBytes);
}

} //namespace armnn
