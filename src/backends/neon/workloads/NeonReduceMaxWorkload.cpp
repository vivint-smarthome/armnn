//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

// TODO

#include "NeonReduceMaxWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <arm_compute/runtime/NEON/functions/NEReduceMaxLayer.h>

#include <boost/polymorphic_cast.hpp>

namespace armnn
{

NeonReduceMaxWorkload::NeonReduceMaxWorkload(const ReduceMaxQueueDescriptor& descriptor,
                                         const WorkloadInfo& info)
    : BaseWorkload<ReduceMaxQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReduceMaxWorkload", 1, 1);

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    auto layer = std::make_unique<arm_compute::NEReduceMaxLayer>();

    // TODO: is the axis being passed into here correctly, or not?

    // TODO: check that we are convert/loading the axis correctly
    arm_compute::Coordinates axis;
    if (descriptor.m_Axis != nullptr) {
        // TODO: this probably isn't correct??
        const void *mem = static_cast<const void*>(descriptor.m_Axis->Map(true));
        TensorShape shape = descriptor.m_Axis->GetShape();
        TensorShape stride = descriptor.m_Axis->GetStrides();

        // TODO: make this work for non int32 data types.
        // TODO: make this work for other stride data types.
        // Assuming 1 dimensional axis for now.
        axis.set_num_dimensions(shape[0]);

        //printf("Axis passed through to NeonReduceMaxWorkload Coordinate: [");
        for (int i = 0; i < shape[0]; i++) {
            axis[i] = *static_cast<const int32_t*>(mem+i*stride[0]);
            //printf(" %d", axis[i]);
        }
        //printf(" ]\n");
        descriptor.m_Axis->Unmap();
    }

    layer->configure(&input, axis, descriptor.m_KeepDims, &output);
    m_Layer.reset(layer.release());
}

void NeonReduceMaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonReduceMaxWorkload_Execute");
    m_Layer->run();
    // print output
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    /*
    std::cout << "Output from NeonReduceMaxWorkload" << std::endl;
    arm_compute::IOFormatInfo iofmt;
    output.print(std::cout, iofmt);
    */
}

} //namespace armnn
