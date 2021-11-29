//
// Created by perseusdg on 11/30/21.
//

#include "TRTDequantizeLayer.h"

int32_t dequantize_layer_get_axis(nvinfer1::IDequantizeLayer *layer){
    return (layer->getAxis());
}

void dequantize_layer_set_axis(nvinfer1::IDequantizeLayer *layer,int32_t axis){
    layer->setAxis(axis);
}