//
// Created by perseusdg on 11/18/21.
//
#include "TRTPaddingLayer.h"

void padding_layer_set_pre_padding(nvinfer1::IPaddingLayer *layer,nvinfer1::Dims padding){
    layer->setPrePaddingNd(padding);
}

nvinfer1::Dims padding_layer_get_pre_padding(nvinfer1::IPaddingLayer* layer){
    return (layer->getPostPaddingNd());
}

void padding_layer_set_post_padding(nvinfer1::IPaddingLayer* layer,nvinfer1::Dims padding){
    layer->setPostPaddingNd(padding);
}

nvinfer1::Dims padding_layer_get_post_padding(nvinfer1::IPaddingLayer* layer){
    return (layer->getPostPaddingNd());
}