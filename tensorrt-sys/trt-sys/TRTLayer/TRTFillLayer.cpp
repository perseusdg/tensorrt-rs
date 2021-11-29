//
// Created by perseusdg on 11/30/21.
//

#include "TRTFillLayer.h"

void fill_layer_set_dimensions(nvinfer1::IFillLayer *layer,nvinfer1::Dims dimensions){
    layer->setDimensions(dimensions);
}

nvinfer1::Dims fill_layer_get_dimensions(nvinfer1::IFillLayer *layer){
    return (layer->getDimensions());
}

void fill_layer_set_operation(nvinfer1::IFillLayer *layer,FillOperation_t operation){
    layer->setOperation(static_cast<nvinfer1::FillOperation>(operation));
}

FillOperation_t fill_layer_get_operation(nvinfer1::IFillLayer *layer){
    return static_cast<FillOperation_t>(layer->getOperation());
}

void fill_layer_set_alpha(nvinfer1::IFillLayer *layer,double alpha){
    layer->setAlpha(alpha);
}

double fill_layer_get_alpha(nvinfer1::IFillLayer *layer){
    return (layer->getAlpha());
}

void fill_layer_set_beta(nvinfer1::IFillLayer *layer,double beta){
    layer->setBeta(beta);
}

double fill_layer_get_beta(nvinfer1::IFillLayer *layer){
    return (layer->getBeta());
}

void fill_layer_set_input(nvinfer1::IFillLayer *layer,int32_t index,nvinfer1::ITensor *tensor){
    layer->setInput(index,*tensor);
}