//
// Created by perseusdg on 11/29/21.
//

#include "TRTConstantLayer.h"

void constant_layer_set_weights(nvinfer1::IConstantLayer *layer,nvinfer1::Weights weights){
    layer->setWeights(weights);
}

nvinfer1::Weights constant_layer_get_weights(nvinfer1::IConstantLayer *layer){
    return(layer->getWeights());
}

void constant_layer_set_dimensions(nvinfer1::IConstantLayer *layer,nvinfer1::Dims dimension){
    layer->setDimensions(dimension);
}

nvinfer1::Dims constant_layer_get_dimensions(nvinfer1::IConstantLayer *layer){
    return(layer->getDimensions());
}