//
// Created by perseusdg on 11/18/21.
//

#ifndef LIBTRT_TRTSCALELAYER_H
#define LIBTRT_TRTSCALELAYER_H

#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"

void scale_layer_set_mode(nvinfer1::IScaleLayer* layer,ScaleMode_t mode);
ScaleMode_t scale_layer_get_mode(nvinfer1::IScaleLayer* layer);
void scale_layer_set_shift(nvinfer1::IScaleLayer* layer,nvinfer1::Weights shift);
nvinfer1::Weights scale_layer_get_shift(nvinfer1::IScaleLayer* layer);
void scale_layer_set_power(nvinfer1::IScaleLayer* layer,nvinfer1::Weights power);
nvinfer1::Weights scale_layer_get_power(nvinfer1::IScaleLayer* layer);
int32_t scale_layer_get_channel_axis(nvinfer1::IScaleLayer* layer);
void scale_layer_set_channel_axis(nvinfer1::IScaleLayer* layer,int32_t channelAxis);

#endif //LIBTRT_TRTSCALELAYER_H
