//
// Created by perseusdg on 11/18/21.
//

#ifndef LIBTRT_TRTRESIZELAYER_H
#define LIBTRT_TRTRESIZELAYER_H
#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"


void resize_layer_set_output_dimensions(nvinfer1::IResizeLayer* layer,nvinfer1::Dims dimensions);
nvinfer1::Dims resize_layer_get_output_dimensions(nvinfer1::IResizeLayer* layer);

#endif //LIBTRT_TRTRESIZELAYER_H
