//
// Created by perseusdg on 28/11/21.
//

#include "TRTConcatenationLayer.h"

void concatenation_layer_set_axis(nvinfer1::IConcatenationLayer *layer,int32_t axis){
    layer->setAxis(axis);
}

int32_t concatenation_layer_get_axis(nvinfer1::IConcatenationLayer *layer){
    return(layer->getAxis());
}