//
// Created by perseusdg on 11/18/21.
//

#ifndef LIBTRT_TRTQUANTIZELAYER_H
#define LIBTRT_TRTQUANTIZELAYER_H
#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"

void quantize_layer_set_axis(nvinfer1::IQuantizeLayer* layer,int32_t axis);
int32_t quantize_layer_get_axis(nvinfer1::IQuantizeLayer* layer);

#endif //LIBTRT_TRTQUANTIZELAYER_H
