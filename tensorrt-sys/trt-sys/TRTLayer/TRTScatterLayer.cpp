//
// Created by perseusdg on 28/11/21.
//

#include "TRTScatterLayer.h"

void scatter_layer_set_mode(nvinfer1::IScatterLayer *layer,ScatterMode_t mode){
    layer->setMode(static_cast<nvinfer1::ScatterMode>(mode));
}

ScatterMode_t scatter_layer_get_mode(nvinfer1::IScatterLayer *layer){
    return (static_cast<ScatterMode_t>(layer->getMode()));
}

void scatter_layer_set_axis(nvinfer1::IScatterLayer *layer,int32_t axis){
    layer->setAxis(axis);
}

int32_t scatter_layer_get_axis(nvinfer1::IScatterLayer *layer){
    return(layer->getAxis());
}