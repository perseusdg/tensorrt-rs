//
// Created by perseusdg on 28/11/21.
//

#ifndef LIBTRT_TRTSCATTERLAYER_H
#define LIBTRT_TRTSCATTERLAYER_H


#include "../TRTEnums.h"
#include "TRTLayer.h"

void scatter_layer_set_mode(nvinfer1::IScatterLayer *layer,ScatterMode_t mode);
ScatterMode_t scatter_layer_get_mode(nvinfer1::IScatterLayer *layer);
void scatter_layer_set_axis(nvinfer1::IScatterLayer *layer,int32_t axis);
int32_t scatter_layer_get_axis(nvinfer1::IScatterLayer *layer);


#endif //LIBTRT_TRTSCATTERLAYER_H
