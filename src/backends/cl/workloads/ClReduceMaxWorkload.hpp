//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/CL/CLFunctions.h>

namespace armnn
{

class ClReduceMaxWorkload : public BaseWorkload<ReduceMaxQueueDescriptor>
{
public:
    ClReduceMaxWorkload(const ReduceMaxQueueDescriptor& descriptor, const WorkloadInfo& info);

    void Execute() const override;

private:
    mutable arm_compute::CLReduceMaxLayer m_Layer;
};

} //namespace armnn
