//
// Created by perseusdg on 28/11/21.
//

#ifndef LIBTRT_TRTSLICELAYER_H
#define LIBTRT_TRTSLICELAYER_H
#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"

void slice_layer_set_start(nvinfer1::ISliceLayer *layer,nvinfer1::Dims start);
nvinfer1::Dims slice_layer_get_start(nvinfer1::ISliceLayer *layer);
void slice_layer_set_size(nvinfer1::ISliceLayer *layer,nvinfer1::Dims size);
nvinfer1::Dims slice_layer_get_size(nvinfer1::ISliceLayer *layer);
void slice_layer_set_stride(nvinfer1::ISliceLayer *layer,nvinfer1::Dims stride);
nvinfer1::Dims slice_layer_get_stride(nvinfer1::ISliceLayer *layer);
void slice_layer_set_mode(nvinfer1::ISliceLayer *layer,SliceMode_t sliceMode);
SliceMode_t slice_layer_get_mode(nvinfer1::ISliceLayer* layer);
void slice_layer_set_input(nvinfer1::ISliceLayer *layer,int32_t index,nvinfer1::ITensor *tensor);





#endif //LIBTRT_TRTSLICELAYER_H
