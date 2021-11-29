//
// Created by perseusdg on 11/30/21.
//

#ifndef LIBTRT_TRTDEQUANTIZELAYER_H
#define LIBTRT_TRTDEQUANTIZELAYER_H


#include "../TRTEnums.h"
#include "TRTLayer.h"

int32_t dequantize_layer_get_axis(nvinfer1::IDequantizeLayer *layer);
void dequantize_layer_set_axis(nvinfer1::IDequantizeLayer *layer,int32_t axis);


#endif //LIBTRT_TRTDEQUANTIZELAYER_H
