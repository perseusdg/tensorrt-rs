//
// Created by perseusdg on 28/11/21.
//

#ifndef TENSORRT_RS_TRTCONCATENATIONLAYER_H
#define TENSORRT_RS_TRTCONCATENATIONLAYER_H


#include <NvInfer.h>
#include "TRTLayer.h"
void concatenation_layer_set_axis(nvinfer1::IConcatenationLayer *layer,int32_t axis);
int32_t concatenation_layer_get_axis(nvinfer1::IConcatenationLayer *layer);

#endif //TENSORRT_RS_TRTCONCATENATIONLAYER_H
