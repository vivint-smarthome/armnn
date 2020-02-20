//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>

namespace armnn
{

class RefReduceMaxWorkload : public BaseWorkload<ReduceMaxQueueDescriptor>
{
public:
    using BaseWorkload<ReduceMaxQueueDescriptor>::BaseWorkload;
    virtual void Execute() const override;
};

} //namespace armnn