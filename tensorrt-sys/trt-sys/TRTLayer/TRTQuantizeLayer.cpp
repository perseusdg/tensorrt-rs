//
// Created by perseusdg on 11/18/21.
//

#include "TRTQuantizeLayer.h"

void quantize_layer_set_axis(nvinfer1::IQuantizeLayer* layer,int32_t axis){
    layer->setAxis(axis);
}
int32_t quantize_layer_get_axis(nvinfer1::IQuantizeLayer* layer){
    return (layer->getAxis());
}