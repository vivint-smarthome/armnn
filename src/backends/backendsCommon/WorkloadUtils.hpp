//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CpuTensorHandle.hpp"
#include "ITensorHandle.hpp"

#include <armnn/Tensor.hpp>

#include <Half.hpp>
#include <Permute.hpp>
#include <Profiling.hpp>
#include <arm_compute/core/Dimensions.h>

#include <boost/cast.hpp>

namespace armnn
{
namespace
{

template <typename ArrayType, typename Arg>
void AssignValues(unsigned int num, unsigned int& idx, const ArrayType& array, Arg& arg)
{
    if (idx >= num)
    {
        return;
    }

    arg = array[(num - 1) - idx];
    idx++;
}

template <typename T, typename ArrayType, typename... Args>
void AssignValues(unsigned int num, unsigned int idx, const ArrayType& array, T& assignee, Args&... args)
{
    AssignValues(num, idx, array, assignee);

    AssignValues(num, idx, array, args...);
}

}    // anonymous namespace

//template <typename CopyFunc>
//void CopyTensorContentsGeneric(const ITensorHandle* srcTensor, ITensorHandle* dstTensor, CopyFunc copy)
//{
//    // For ease of understanding, names are assigned to the dimensions
//    // of the tensor as if NHWC, however this routine works with any 5D tensor
//    static_assert(MaxNumOfTensorDimensions == 5, "Please update CopyTensorContents");
//
//    TensorShape srcStrides      = srcTensor->GetStrides();
//    const TensorShape& srcShape = srcTensor->GetShape();
//    TensorShape dstStrides      = dstTensor->GetStrides();
//    const TensorShape& dstShape = dstTensor->GetShape();
//
//    size_t srcDepth    = 1;
//    size_t srcBatches  = 1;
//    size_t srcHeight   = 1;
//    size_t srcWidth    = 1;
//    size_t srcChannels = 1;
//    AssignValues(srcShape.GetNumDimensions(),
//                 0,
//                 srcShape,
//                 srcChannels,
//                 srcWidth,
//                 srcHeight,
//                 srcBatches,
//                 srcDepth);
//
//    size_t srcDepthStride   = 0;
//    size_t srcBatchStride   = 0;
//    size_t srcHeightStride  = 0;
//    size_t srcWidthStride   = 0;
//    size_t srcChannelStride = 0;
//    AssignValues(srcStrides.GetNumDimensions(),
//                 0,
//                 srcStrides,
//                 srcChannelStride,
//                 srcWidthStride,
//                 srcHeightStride,
//                 srcBatchStride,
//                 srcDepthStride);
//
//    size_t dstDepth    = 1;
//    size_t dstBatches  = 1;
//    size_t dstHeight   = 1;
//    size_t dstWidth    = 1;
//    size_t dstChannels = 1;
//    AssignValues(dstShape.GetNumDimensions(),
//                 0,
//                 dstShape,
//                 dstChannels,
//                 dstWidth,
//                 dstHeight,
//                 dstBatches,
//                 dstDepth);
//
//    size_t dstDepthStride   = 0;
//    size_t dstBatchStride   = 0;
//    size_t dstHeightStride  = 0;
//    size_t dstWidthStride   = 0;
//    size_t dstChannelStride = 0;
//    AssignValues(dstStrides.GetNumDimensions(),
//                 0,
//                 dstStrides,
//                 dstChannelStride,
//                 dstWidthStride,
//                 dstHeightStride,
//                 dstBatchStride,
//                 dstDepthStride);
//
//    const unsigned char* srcData;
//    unsigned char* dstData;
//    {
//        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Synchronize buffers");
//        srcData = static_cast<const uint8_t*>(srcTensor->Map());
//        dstData = static_cast<uint8_t*>(dstTensor->Map());
//    }
//
//    size_t copyLength  = std::min(srcChannels*srcChannelStride, dstChannels*dstChannelStride);
//    size_t copyWidth   = std::min(srcWidth, dstWidth);
//    size_t copyHeight  = std::min(srcHeight, dstHeight);
//    size_t copyBatches = std::min(srcBatches, dstBatches);
//    size_t copyDepth   = std::min(srcDepth, dstDepth);
//
//    // Coalesce inner dimensions where possible
//    // to reduce overheard calling copy() and to
//    // allow for memory bandwidth optimisations
//    if (copyLength == srcWidthStride &&
//        copyLength == dstWidthStride)
//    {
//        // There is no special padding between rows,
//        // and sizes are compatible, so copy whole rows
//        copyLength *= copyWidth;
//        copyWidth = 1;
//
//        if (copyLength == srcHeightStride &&
//            copyLength == dstHeightStride)
//        {
//            // There is no special padding between batches
//            // and sizes are compatible so copy whole batches
//            copyLength *= copyHeight;
//            copyHeight = 1;
//        }
//    }
//
//    for (unsigned int d = 0; d < copyDepth; ++d)
//    {
//        auto srcPtrDepth = srcData;
//        auto dstPtrDepth = dstData;
//        for (unsigned int b = 0; b < copyBatches; ++b)
//        {
//            auto srcPtrBatch = srcData;
//            auto dstPtrBatch = dstData;
//            for (unsigned int h = 0; h < copyHeight; ++h)
//            {
//                auto srcPtrChannel = srcData;
//                auto dstPtrChannel = dstData;
//                for (unsigned int w = 0; w < copyWidth; ++w)
//                {
//                    copy(dstData, srcData, copyLength);
//                    dstData += dstWidthStride;
//                    srcData += srcWidthStride;
//                }
//                dstData += (static_cast<long>(dstHeightStride) - (dstData - dstPtrChannel));
//                srcData += (static_cast<long>(srcHeightStride) - (srcData - srcPtrChannel));
//            }
//            dstData += (static_cast<long>(dstBatchStride) - (dstData - dstPtrBatch));
//            srcData += (static_cast<long>(srcBatchStride) - (srcData - srcPtrBatch));
//        }
//        dstData += (static_cast<long>(dstDepthStride) - (dstData - dstPtrDepth));
//        srcData += (static_cast<long>(srcDepthStride) - (srcData - srcPtrDepth));
//    }
//
//    srcTensor->Unmap();
//    dstTensor->Unmap();
//}

static bool increment_pos(std::vector<size_t> &pos, size_t iter_dimension, TensorShape const &srcDimensionSize, TensorShape const &dstDimensionSize) {
    pos[iter_dimension]++;

    for (ssize_t i = iter_dimension; i >= 1; i--) {
        size_t dimensionSize = std::min(dstDimensionSize[i], srcDimensionSize[i]);

        // TODO: should this be >= instead of > ?
        if (pos[i] >= dimensionSize) {
            pos[i] = 0;
            pos[i-1]++;
        }
    }

    return pos[0] >= std::min(dstDimensionSize[0], srcDimensionSize[0]);
}

static size_t dim_index(std::vector<size_t> const &pos, TensorShape const &dimensionStride) {
    size_t idx = 0;
    for (size_t i = 0; i < pos.size(); i++) {
        idx += pos[i] * dimensionStride[i];
    }
    return idx;
}

// CopyFunc:
// copy(dst, src, bytes)
template <typename CopyFunc>
void XCopyTensorContentsGeneric(const ITensorHandle* srcTensor, ITensorHandle* dstTensor, CopyFunc copy)
{
    //std::cout << "me CopyTensorContentsGeneric Function: is not complete" << std::endl;

    TensorShape srcStrides      = srcTensor->GetStrides();
    const TensorShape& srcShape = srcTensor->GetShape();
    TensorShape dstStrides      = dstTensor->GetStrides();
    const TensorShape& dstShape = dstTensor->GetShape();

    const unsigned char* srcData;
    unsigned char* dstData;
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Synchronize buffers");
        srcData = static_cast<const uint8_t*>(srcTensor->Map());
        dstData = static_cast<uint8_t*>(dstTensor->Map());
    }

    /*
    for (int i = 0; i < srcStrides.GetNumDimensions(); i++) {
        if (srcStrides[i] != dstStrides[i]) {
            std::cout << "Stride " << i << " srcStride: " << srcStrides[i] << " dstStride: " << dstStrides[i] << std::endl;
        }
        //std::cout << "Stride[" << i << "]: " << srcStrides[i] << std::endl;
    }
    for (int i = 0; i < srcShape.GetNumDimensions(); i++) {
        if (srcShape[i] != dstShape[i]) {
            std::cout << "Shape " << i << " srcShape: " << srcShape[i] << " dstShape: " << dstShape[i] << std::endl;
        }
        //std::cout << "Shape[" << i << "]: " << srcShape[i] << std::endl;
    }
    */

    std::vector<size_t> pos(srcStrides.GetNumDimensions());
    std::fill(pos.begin(), pos.end(), 0);

    // Coalesce inner dimensions to save calls to copy()
    size_t iter_dim = pos.size()-1;

    for (int i = pos.size()-1; i >= 0; i--) {
        if (srcStrides[i] != dstStrides[i]
            || srcShape[i] != dstShape[i]) {
            break;
        } else {
            iter_dim = i;
        }
    }

    // std::cout << "chosen iter_dim " << iter_dim << std::endl;

    size_t copy_calls = 0;
    bool done = false;
    size_t copy_bytes = std::min(srcStrides[iter_dim], dstStrides[iter_dim]);

    while (!done) {
        size_t src_idx = dim_index(pos, srcStrides);
        size_t dst_idx = dim_index(pos, dstStrides);

        /*
        if (true || srcShape[1] == 2391) {
            std::cout << "pos:";
            for (auto it = pos.begin(); it != pos.end(); ++it) {
                std::cout << " " << *it;
            }
            std::cout << " src_idx: " << src_idx << " dst_idx: " << dst_idx << std::endl;
        }
        */

        copy(dstData + dst_idx, srcData + src_idx, copy_bytes);
        copy_calls++;
        done = increment_pos(pos, iter_dim, srcShape, dstShape);
    }
    // std::cout << "Executed " << copy_calls << " Copy Calls" << std::endl;

    srcTensor->Unmap();
    dstTensor->Unmap();
}

template <typename CopyFunc>
void CopyTensorContentsGeneric(const ITensorHandle* srcTensor, ITensorHandle* dstTensor, CopyFunc copy)
{
    // For ease of understanding, names are assigned to the dimensions
    // of the tensor as if NHWC, however this routine works with any 6D tensor
    static_assert(MaxNumOfTensorDimensions == 6, "Please update CopyTensorContents");

    TensorShape srcStrides      = srcTensor->GetStrides();
    const TensorShape& srcShape = srcTensor->GetShape();
    TensorShape dstStrides      = dstTensor->GetStrides();
    const TensorShape& dstShape = dstTensor->GetShape();

    size_t srcI0 = 1;
    size_t srcDepth    = 1;
    size_t srcBatches  = 1;
    size_t srcHeight   = 1;
    size_t srcWidth    = 1;
    size_t srcChannels = 1;

    AssignValues(srcShape.GetNumDimensions(),
                 0,
                 srcShape,
                 srcChannels,
                 srcWidth,
                 srcHeight,
                 srcBatches,
                 srcDepth,
                 srcI0
                 );

    size_t srcI0Stride = 0;
    size_t srcDepthStride   = 0;
    size_t srcBatchStride   = 0;
    size_t srcHeightStride  = 0;
    size_t srcWidthStride   = 0;
    size_t srcChannelStride = 0;
    AssignValues(srcStrides.GetNumDimensions(),
                 0,
                 srcStrides,
                 srcChannelStride,
                 srcWidthStride,
                 srcHeightStride,
                 srcBatchStride,
                 srcDepthStride,
                 srcI0Stride
                 );

    size_t dstI0 = 1;
    size_t dstDepth    = 1;
    size_t dstBatches  = 1;
    size_t dstHeight   = 1;
    size_t dstWidth    = 1;
    size_t dstChannels = 1;
    AssignValues(dstShape.GetNumDimensions(),
                 0,
                 dstShape,
                 dstChannels,
                 dstWidth,
                 dstHeight,
                 dstBatches,
                 dstDepth,
                 dstI0
                 );

    size_t dstI0Stride = 0;
    size_t dstDepthStride   = 0;
    size_t dstBatchStride   = 0;
    size_t dstHeightStride  = 0;
    size_t dstWidthStride   = 0;
    size_t dstChannelStride = 0;
    AssignValues(dstStrides.GetNumDimensions(),
                 0,
                 dstStrides,
                 dstChannelStride,
                 dstWidthStride,
                 dstHeightStride,
                 dstBatchStride,
                 dstDepthStride,
                 dstI0Stride);

    const unsigned char* srcData;
    unsigned char* dstData;
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "Synchronize buffers");
        srcData = static_cast<const uint8_t*>(srcTensor->Map());
        dstData = static_cast<uint8_t*>(dstTensor->Map());
    }

    size_t copyLength  = std::min(srcChannels*srcChannelStride, dstChannels*dstChannelStride);
    size_t copyWidth   = std::min(srcWidth, dstWidth);
    size_t copyHeight  = std::min(srcHeight, dstHeight);
    size_t copyBatches = std::min(srcBatches, dstBatches);
    size_t copyDepth   = std::min(srcDepth, dstDepth);
    size_t copyI0 = std::min(srcI0, dstI0);

    // Coalesce inner dimensions where possible
    // to reduce overheard calling copy() and to
    // allow for memory bandwidth optimisations
    if (copyLength == srcWidthStride &&
        copyLength == dstWidthStride)
    {
        // There is no special padding between rows,
        // and sizes are compatible, so copy whole rows
        copyLength *= copyWidth;
        copyWidth = 1;

        if (copyLength == srcHeightStride &&
            copyLength == dstHeightStride)
        {
            // There is no special padding between batches
            // and sizes are compatible so copy whole batches
            copyLength *= copyHeight;
            copyHeight = 1;
        }
    }

    for (unsigned int i0 = 0; i0 < copyI0; ++i0) 
    {
        auto srcPtrI0 = srcData;
        auto dstPtrI0 = dstData;
        for (unsigned int d = 0; d < copyDepth; ++d)
        {
            auto srcPtrDepth = srcData;
            auto dstPtrDepth = dstData;
            for (unsigned int b = 0; b < copyBatches; ++b)
            {
                auto srcPtrBatch = srcData;
                auto dstPtrBatch = dstData;
                for (unsigned int h = 0; h < copyHeight; ++h)
                {
                    auto srcPtrChannel = srcData;
                    auto dstPtrChannel = dstData;
                    for (unsigned int w = 0; w < copyWidth; ++w)
                    {
                        copy(dstData, srcData, copyLength);
                        dstData += dstWidthStride;
                        srcData += srcWidthStride;
                    }
                    dstData += (static_cast<long>(dstHeightStride) - (dstData - dstPtrChannel));
                    srcData += (static_cast<long>(srcHeightStride) - (srcData - srcPtrChannel));
                }
                dstData += (static_cast<long>(dstBatchStride) - (dstData - dstPtrBatch));
                srcData += (static_cast<long>(srcBatchStride) - (srcData - srcPtrBatch));
            }
            dstData += (static_cast<long>(dstDepthStride) - (dstData - dstPtrDepth));
            srcData += (static_cast<long>(srcDepthStride) - (srcData - srcPtrDepth));
        }
        dstData += (static_cast<long>(dstI0Stride) - (dstData - dstPtrI0));
        srcData += (static_cast<long>(srcI0Stride) - (srcData - srcPtrI0));
    }

    srcTensor->Unmap();
    dstTensor->Unmap();
}

template <typename SrcTensorHandleType, typename DstTensorHandleType, typename DescriptorType>
void GatherTensorHandlePairs(const DescriptorType& descriptor,
                             std::vector<std::pair<SrcTensorHandleType*, DstTensorHandleType*>>& tensorHandlePairs)
{
    const unsigned int numInputs = static_cast<unsigned int>(descriptor.m_Inputs.size());
    tensorHandlePairs.reserve(numInputs);

    for (unsigned int i = 0; i < numInputs; ++i)
    {
        SrcTensorHandleType* const srcTensorHandle =
            boost::polymorphic_downcast<SrcTensorHandleType*>(descriptor.m_Inputs[i]);
        DstTensorHandleType* const dstTensorHandle =
            boost::polymorphic_downcast<DstTensorHandleType*>(descriptor.m_Outputs[i]);

        tensorHandlePairs.emplace_back(srcTensorHandle, dstTensorHandle);
    }
}

int32_t ConvertMaskToACLFormat(int32_t mask, int32_t numDim);

armnn::ConstTensor PermuteTensor(const ConstCpuTensorHandle* tensor,
                                 const PermutationVector& permutationVector,
                                 void* permuteBuffer);

void ReshapeWeightsForAcl(TensorInfo& weightInfo, DataLayout dataLayout);

TensorInfo ConvertWeightTensorInfoFromArmnnToAcl(const TensorInfo& weightInfo, DataLayout dataLayout);

armnn::ConstTensor ConvertWeightTensorFromArmnnToAcl(const ConstCpuTensorHandle* weightTensor,
                                                     DataLayout dataLayout,
                                                     void* permuteBuffer);

}  //namespace armnn
