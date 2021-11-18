//
// Created by perseusdg on 11/18/21.
//

#ifndef LIBTRT_TRTREDUCELAYER_H
#define LIBTRT_TRTREDUCELAYER_H
#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"

void reduce_layer_set_operation(nvinfer1::IReduceLayer* layer,ReduceOperation_t op);
ReduceOperation_t reduce_layer_get_operation(nvinfer1::IReduceLayer *layer);
void reduce_layer_set_reduce_axis(nvinfer1::IReduceLayer* layer,uint32_t reduceAxes);
uint32_t reduce_layer_get_reduce_axis(nvinfer1::IReduceLayer* layer);
void reduce_layer_set_keep_dimensions(nvinfer1::IReduceLayer* layer,bool keep);
bool reduce_layer_get_keep_dimensions(nvinfer1::IReduceLayer* layer);

#endif //LIBTRT_TRTREDUCELAYER_H
