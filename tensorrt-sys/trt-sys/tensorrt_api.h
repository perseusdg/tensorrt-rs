//
// Created by mason on 8/25/19.
//

#ifndef TENSORRT_SYS_TENSORRT_API_H
#define TENSORRT_SYS_TENSORRT_API_H
#include <NvInfer.h>

#include "TRTEnums.h"
#include "TRTLogger/TRTLogger.h"
#include "TRTRuntime/TRTRuntime.h"
#include "TRTCudaEngine/TRTCudaEngine.h"
#include "TRTContext/TRTContext.h"
#include "TRTUffParser/TRTUffParser.h"
#include "TRTOnnxParser/TRTOnnxParser.h"
#include "TRTDims/TRTDims.h"
#include "TRTBuilder/TRTBuilder.h"
#include "TRTNetworkDefinition/TRTNetworkDefinition.h"
#include "TRTHostMemory/TRTHostMemory.h"
#include "TRTProfiler/TRTProfiler.h"
#include "TRTTensor/TRTTensor.h"
#include "TRTLayer/TRTLayer.h"
#include "TRTLayer/TRTElementWiseLayer.h"
#include "TRTLayer/TRTGatherLayer.h"
#include "TRTLayer/TRTActivationLayer.h"
#include "TRTLayer/TRTPoolingLayer.h"
#include "TRTLayer/TRTPaddingLayer.h"
#include "TRTLayer/TRTQuantizeLayer.h"
#include "TRTLayer/TRTReduceLayer.h"
#include "TRTLayer/TRTResizeLayer.h"
#include "TRTLayer/TRTScaleLayer.h"
#include "TRTLayer/TRTConcatenationLayer.h"
#include "TRTLayer/TRTScatterLayer.h"
#include "TRTWeights/TRTWeights.h"
#include "TRTLayer/TRTSliceLayer.h"
#include "TRTLayer/TRTConstantLayer.h"
#include "TRTLayer/TRTDeconvolutionLayer.h"
#include "TRTLayer/TRTConvolutionLayer.h"
#include "TRTLayer/TRTDequantizeLayer.h"
#include "TRTLayer/TRTFillLayer.h"

#endif //TENSORRT_SYS_TENSORRT_API_H
