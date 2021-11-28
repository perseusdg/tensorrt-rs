//
// Created by perseusdg on 28/11/21.
//

#include "TRTSliceLayer.h"
void slice_layer_set_start(nvinfer1::ISliceLayer *layer,nvinfer1::Dims start){
    layer->setStart(start);
}

nvinfer1::Dims slice_layer_get_start(nvinfer1::ISliceLayer *layer){
    return(layer->getStart());
}

void slice_layer_set_size(nvinfer1::ISliceLayer *layer,nvinfer1::Dims size){
    layer->setSize(size);
}

nvinfer1::Dims slice_layer_get_size(nvinfer1::ISliceLayer *layer){
    return(layer->getSize());
}

void slice_layer_set_stride(nvinfer1::ISliceLayer *layer,nvinfer1::Dims stride){
    layer->setStride(stride);
}

nvinfer1::Dims slice_layer_get_stride(nvinfer1::ISliceLayer *layer){
    return(layer->getStride());
}

void slice_layer_set_mode(nvinfer1::ISliceLayer *layer,SliceMode_t sliceMode){
    layer->setMode(static_cast<nvinfer1::SliceMode>(sliceMode));
}

SliceMode_t slice_layer_get_mode(nvinfer1::ISliceLayer* layer){
    return(static_cast<SliceMode_t>(layer->getMode()));
}

void slice_layer_set_input(nvinfer1::ISliceLayer *layer,int32_t index,nvinfer1::ITensor *tensor){
    layer->setInput(index,*tensor);
}