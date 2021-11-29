//
// Created by perseusdg on 11/30/21.
//

#include "TRTFullyConnectedLayer.h"

void fully_connected_layer_set_nb_output_channels(nvinfer1::IFullyConnectedLayer *layer,int32_t nbOutputs){
    layer->setNbOutputChannels(nbOutputs);
}

int32_t fully_connected_layer_get_nb_output_channels(nvinfer1::IFullyConnectedLayer *layer){
    return (layer->getNbOutputChannels());
}

void fully_connected_layer_set_kernel_weights(nvinfer1::IFullyConnectedLayer *layer,nvinfer1::Weights weights){
    layer->setKernelWeights(weights);
}

nvinfer1::Weights fully_connected_layer_get_kernel_weights(nvinfer1::IFullyConnectedLayer *layer){
    return (layer->getKernelWeights());
}

void fully_connected_layer_set_bias_weights(nvinfer1::IFullyConnectedLayer *layer,nvinfer1::Weights weights){
    layer->setBiasWeights(weights);
}

nvinfer1::Weights fully_connected_layer_get_bias_weights(nvinfer1::IFullyConnectedLayer *layer){
    return (layer->getBiasWeights());
}
