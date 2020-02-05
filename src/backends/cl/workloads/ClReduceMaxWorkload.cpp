//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClReduceMaxWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClReduceMaxWorkload::ClReduceMaxWorkload(const ReduceMaxQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ReduceMaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClReduceMaxWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_Layer.configure(&input, &output);
}

void ClReduceMaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClReduceMaxWorkload_Execute");
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} //namespace armnn
