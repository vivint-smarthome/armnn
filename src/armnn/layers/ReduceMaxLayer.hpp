//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

// TODO

namespace armnn
{
class ScopedCpuTensorHandle;


/// This layer represents a reducemax operation.
class ReduceMaxLayer : public LayerWithParameters<ReduceMaxDescriptor>
{
public:
    /// A unique pointer to store axis values
    std::unique_ptr<ScopedCpuTensorHandle> m_Axis;

    /// Makes a workload for the ReduceMax type.
    /// @param [in] graph The graph where this layer can be found.
    /// @param [in] factory The workload factory which will create the workload.
    /// @return A pointer to the created workload, or nullptr if not created.
    virtual std::unique_ptr<IWorkload> CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const override;

    /// Creates a dynamically-allocated copy of this layer.
    /// @param [in] graph The graph into which this layer is being cloned.
    ReduceMaxLayer* Clone(Graph& graph) const override;

    /// Check if the input tensor shape(s)
    /// will lead to a valid configuration of @ref ReduceMaxLayer.
    void ValidateTensorShapesFromInputs() override;

    /// By default returns inputShapes if the number of inputs are equal to number of outputs,
    /// otherwise infers the output shapes from given input shapes and layer properties.
    /// @param [in] inputShapes The input shapes layer has.
    /// @return A vector to the inferred output shape.
    std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const override;

    void Accept(ILayerVisitor& visitor) const override;

protected:
    /// Constructor to create a ReduceMaxLayer.
    /// @param [in] param ReduceMaxDescriptor to configure the reduce max operation.
    /// @param [in] name Optional name for the layer.
    ReduceMaxLayer(const ReduceMaxDescriptor& desc, const char* name);

    /// Default destructor
    ~ReduceMaxLayer() = default;

};

} // namespace
