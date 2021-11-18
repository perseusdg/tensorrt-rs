//
// Created by perseusdg on 11/18/21.
//

#include "TRTScaleLayer.h"

void scale_layer_set_mode(nvinfer1::IScaleLayer* layer,ScaleMode_t mode) {
    layer->setMode(static_cast<nvinfer1::ScaleMode>(mode));
}

ScaleMode_t scale_layer_get_mode(nvinfer1::IScaleLayer* layer){
    return(static_cast<ScaleMode_t>(layer->getMode()));
}

void scale_layer_set_shift(nvinfer1::IScaleLayer* layer,nvinfer1::Weights shift) {
    layer->setShift(shift);
}

nvinfer1::Weights scale_layer_get_shift(nvinfer1::IScaleLayer* layer) {
    return (layer->getScale());
}

void scale_layer_set_power(nvinfer1::IScaleLayer* layer,nvinfer1::Weights power) {
    layer->setPower(power);
}

nvinfer1::Weights scale_layer_get_power(nvinfer1::IScaleLayer* layer) {
    return (layer->getPower());
}

int32_t scale_layer_get_channel_axis(nvinfer1::IScaleLayer* layer) {
    return(layer->getChannelAxis());
}

void scale_layer_set_channel_axis(nvinfer1::IScaleLayer* layer,int32_t channelAxis) {
    layer->setChannelAxis(channelAxis);
}

