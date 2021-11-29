//
// Created by perseusdg on 11/29/21.
//

#ifndef LIBTRT_TRTCONSTANTLAYER_H
#define LIBTRT_TRTCONSTANTLAYER_H

#include "../TRTEnums.h"
#include "TRTLayer.h"

void constant_layer_set_weights(nvinfer1::IConstantLayer *layer,nvinfer1::Weights weights);
nvinfer1::Weights constant_layer_get_weights(nvinfer1::IConstantLayer *layer);
void constant_layer_set_dimensions(nvinfer1::IConstantLayer *layer,nvinfer1::Dims dimension);
nvinfer1::Dims constant_layer_get_dimensions(nvinfer1::IConstantLayer *layer);


#endif //LIBTRT_TRTCONSTANTLAYER_H
