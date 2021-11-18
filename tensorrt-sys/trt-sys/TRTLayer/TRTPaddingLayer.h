//
// Created by perseusdg on 11/18/21.
//

#ifndef TENSORRT_RS_TRTPADDINGLAYER_H
#define TENSORRT_RS_TRTPADDINGLAYER_H
#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"

void padding_layer_set_pre_padding(nvinfer1::IPaddingLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims padding_layer_get_pre_padding(nvinfer1::IPaddingLayer* layer);
void padding_layer_set_post_padding(nvinfer1::IPaddingLayer* layer,nvinfer1::Dims padding);
nvinfer1::Dims padding_layer_get_post_padding(nvinfer1::IPaddingLayer* layer);




#endif //TENSORRT_RS_TRTPADDINGLAYER_H
