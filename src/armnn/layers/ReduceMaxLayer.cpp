//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ReduceMaxLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <string>
#include <DataLayoutIndexed.hpp>

using namespace armnnUtils;

// TODO

namespace armnn
{

ReduceMaxLayer::ReduceMaxLayer(const ReduceMaxDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ReduceMax, param, name)
{
}

std::unique_ptr<IWorkload> ReduceMaxLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ReduceMaxQueueDescriptor descriptor;
    if (m_Param.m_HasAxis) {
        descriptor.m_Axis = m_Axis.get();
        TensorShape shape = m_Axis->GetShape();
        TensorShape stride = m_Axis->GetStrides();
        const int32_t *mem = static_cast<const int32_t*>(descriptor.m_Axis->Map(true));
        descriptor.m_Axis->Unmap();
    }
    descriptor.m_KeepDims = m_Param.m_KeepDims;
    return factory.CreateReduceMax(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ReduceMaxLayer* ReduceMaxLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<ReduceMaxLayer>(graph, m_Param, GetName());
    if (layer->m_Param.m_HasAxis) {
        layer->m_Axis = m_Axis ? std::make_unique<ScopedCpuTensorHandle>(*m_Axis) : nullptr;
    }
    return std::move(layer);
}

// TODO: find a way to preserve negative axis inputs passed in
std::vector<TensorShape> ReduceMaxLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    // Yuck! Temp hack for now, because we need the actual axis values to determine the shape,
    // not just the shape of the axis parameters
    return std::vector<TensorShape>({GetOutputSlot(0).GetTensorInfo().GetShape()});

    #if 0
    BOOST_ASSERT(inputShapes.size() == 1 || inputShapes.size() == 2);

    const TensorShape &inputShape = inputShapes[0];
    // const TensorShape axisShape = inputShapes[1];
    TensorShape outputShape;

    std::unordered_set<int> axisSet;
    if (inputShapes.size() == 2) {
        for (int i = 0; i < inputShapes[1].GetNumDimensions(); i++) {
            int axis_param = inputShapes[1][i];
            while (axis_param < 0) {
                axis_param += inputShapes[0].GetNumDimensions();
            }
            axisSet.insert(axis_param);
        }
    }

                GetInputSlot(1).GetConnection()->GetTensorInfo();

    printf("axisSet[");
    for (auto it = axisSet.begin(); it != axisSet.end(); ++it) {
        printf(" %d", *it);
    }
    printf("]\n");
    
    // Output shapes:
    // TODO: how does axis parameter work?
    // keepdims && hasaxis { same ddims, but reduce along given axises}?
    if (m_Param.m_KeepDims && m_Param.m_HasAxis) {
        outputShape = TensorShape(inputShapes[0].GetNumDimensions());
        // If axis includes an input dim, that dim is reduced to 1.
        for(int i = 0; i < inputShapes[0].GetNumDimensions(); i++) {
            printf("i: %d inputShapes[0][i]: %d\n", i, inputShapes[0][i]);
            if (axisSet.find(i) != axisSet.end()) {
                outputShape[i] = 1;
            }
            else {
                outputShape[i] = inputShapes[0][i];
            }
        }
    }
    // keepdims && !hasaxis { same dims, but all 1s }?
    else if (m_Param.m_KeepDims && !m_Param.m_HasAxis) {
        outputShape = TensorShape(inputShapes[0].GetNumDimensions());
        for (int i = 0; i < inputShapes[0].GetNumDimensions(); i++) {
            outputShape[i] = 1;
        }
    }
    // !keepdims && hasaxis { reduce along the given axes }
    else if (!m_Param.m_KeepDims && m_Param.m_HasAxis) {
        throw Exception("ReduceMaxLayer::InferOutputShapes not implemented yet");
    }
    // !keepdims && !hasaxis { 1 }
    else if (!m_Param.m_KeepDims && !m_Param.m_HasAxis) {
        outputShape = TensorShape({1});
    }

    printf("Inferred Output Shape: [");
    for (int i = 0; i < outputShape.GetNumDimensions(); i++) {
        printf(" %d", outputShape[i]);
    }
    printf(" ]\n");

    return std::vector<TensorShape>({ outputShape });
#endif
}

void ReduceMaxLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes;
    
    if (m_Param.m_HasAxis) {
        inferredShapes = InferOutputShapes({
                GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
                m_Axis->GetTensorInfo().GetShape() });
    }
    else {
        inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()});
    }

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ReduceMaxLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ReduceMaxLayer::Accept(ILayerVisitor& visitor) const
{
    Optional<ConstTensor> optionalAxisTensor = EmptyOptional();
    if (GetParameters().m_HasAxis)
    {
        ConstTensor axisTensor(m_Axis->GetTensorInfo(), m_Axis->Map(true));
        optionalAxisTensor = Optional<ConstTensor>(axisTensor);
    }

    visitor.VisitReduceMaxLayer(this, GetParameters(), optionalAxisTensor, GetName());
}

} // namespace armnn
