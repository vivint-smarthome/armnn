//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Abs.hpp"
#include "Activation.hpp"
#include "ArgMinMax.hpp"
#include "BatchNormImpl.hpp"
#include "ConvImpl.hpp"
#include "Concatenate.hpp"
#include "ElementwiseFunction.hpp"
#include "FullyConnected.hpp"
#include "Gather.hpp"
#include "Pooling2d.hpp"
#include "RefAbsWorkload.hpp"
#include "RefActivationWorkload.hpp"
#include "RefArgMinMaxWorkload.hpp"
#include "RefBatchNormalizationWorkload.hpp"
#include "RefBatchToSpaceNdWorkload.hpp"
#include "RefComparisonWorkload.hpp"
#include "RefConvolution2dWorkload.hpp"
#include "RefConstantWorkload.hpp"
#include "RefConcatWorkload.hpp"
#include "RefConvertFp16ToFp32Workload.hpp"
#include "RefConvertFp32ToFp16Workload.hpp"
#include "RefDebugWorkload.hpp"
#include "RefDepthToSpaceWorkload.hpp"
#include "RefDepthwiseConvolution2dWorkload.hpp"
#include "RefDequantizeWorkload.hpp"
#include "RefDetectionPostProcessWorkload.hpp"
#include "RefDequantizeWorkload.hpp"
#include "RefElementwiseWorkload.hpp"
#include "RefFullyConnectedWorkload.hpp"
#include "RefFloorWorkload.hpp"
#include "RefFakeQuantizationFloat32Workload.hpp"
#include "RefGatherWorkload.hpp"
#include "RefInstanceNormalizationWorkload.hpp"
#include "RefL2NormalizationWorkload.hpp"
#include "RefLogSoftmaxWorkload.hpp"
#include "RefLstmWorkload.hpp"
#include "RefMeanWorkload.hpp"
#include "RefNormalizationWorkload.hpp"
#include "RefPooling2dWorkload.hpp"
#include "RefPermuteWorkload.hpp"
#include "RefPadWorkload.hpp"
#include "RefPreluWorkload.hpp"
#include "RefQuantizeWorkload.hpp"
#include "RefReshapeWorkload.hpp"
#include "RefReduceMaxWorkload.hpp"
#include "RefResizeBilinearWorkload.hpp"
#include "RefResizeWorkload.hpp"
#include "RefRsqrtWorkload.hpp"
#include "RefSliceWorkload.hpp"
#include "RefSplitterWorkload.hpp"
#include "RefSoftmaxWorkload.hpp"
#include "RefSpaceToBatchNdWorkload.hpp"
#include "RefStackWorkload.hpp"
#include "RefStridedSliceWorkload.hpp"
#include "RefSpaceToDepthWorkload.hpp"
#include "RefTransposeConvolution2dWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "Resize.hpp"
#include "Softmax.hpp"
#include "Splitter.hpp"
#include "TensorBufferArrayView.hpp"
