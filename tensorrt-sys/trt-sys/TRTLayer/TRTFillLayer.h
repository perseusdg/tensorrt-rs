//
// Created by perseusdg on 11/30/21.
//

#ifndef LIBTRT_TRTFILLLAYER_H
#define LIBTRT_TRTFILLLAYER_H

#include "../TRTEnums.h"
#include "TRTLayer.h"

void fill_layer_set_dimensions(nvinfer1::IFillLayer *layer,nvinfer1::Dims dimensions);
nvinfer1::Dims fill_layer_get_dimensions(nvinfer1::IFillLayer *layer);
void fill_layer_set_operation(nvinfer1::IFillLayer *layer,FillOperation_t operation);
FillOperation_t fill_layer_get_operation(nvinfer1::IFillLayer *layer);
void fill_layer_set_alpha(nvinfer1::IFillLayer *layer,double alpha);
double fill_layer_get_alpha(nvinfer1::IFillLayer *layer);
void fill_layer_set_beta(nvinfer1::IFillLayer *layer,double beta);
double fill_layer_get_beta(nvinfer1::IFillLayer *layer);
void fill_layer_set_input(nvinfer1::IFillLayer *layer,int32_t index,nvinfer1::ITensor *tensor);

#endif //LIBTRT_TRTFILLLAYER_H
