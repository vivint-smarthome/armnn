//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>

#include <arm_compute/runtime/IFunction.h>

#include <memory>

namespace armnn
{

class NeonReduceMaxWorkload : public BaseWorkload<ReduceMaxQueueDescriptor>
{
public:
    NeonReduceMaxWorkload(const ReduceMaxQueueDescriptor& descriptor, const WorkloadInfo& info);

    virtual void Execute() const override;

private:
    std::unique_ptr<arm_compute::IFunction> m_Layer;
};

} //namespace armnn
