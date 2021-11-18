//
// Created by perseusdg on 11/18/21.
//

#include "TRTReduceLayer.h"

void reduce_layer_set_operation(nvinfer1::IReduceLayer* layer,ReduceOperation_t op){
    layer->setOperation(static_cast<nvinfer1::ReduceOperation>(op));
}

ReduceOperation_t reduce_layer_get_operation(nvinfer1::IReduceLayer *layer){
    return(static_cast<ReduceOperation_t>(layer->getOperation()));
}

void reduce_layer_set_reduce_axis(nvinfer1::IReduceLayer* layer,uint32_t reduceAxes){
    layer->setReduceAxes(reduceAxes);
}

uint32_t reduce_layer_get_reduce_axis(nvinfer1::IReduceLayer* layer){
    return(layer->getReduceAxes());
}

void reduce_layer_set_keep_dimensions(nvinfer1::IReduceLayer* layer,bool keep){
    layer->setKeepDimensions(keep);
}

bool reduce_layer_get_keep_dimensions(nvinfer1::IReduceLayer* layer){
    return (layer->getKeepDimensions());
}