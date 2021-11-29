//
// Created by perseusdg on 11/30/21.
//

#ifndef LIBTRT_TRTFULLYCONNECTEDLAYER_H
#define LIBTRT_TRTFULLYCONNECTEDLAYER_H

#include "../TRTEnums.h"
#include "TRTLayer.h"

void fully_connected_layer_set_nb_output_channels(nvinfer1::IFullyConnectedLayer *layer,int32_t nbOutputs);
int32_t fully_connected_layer_get_nb_output_channels(nvinfer1::IFullyConnectedLayer *layer);
void fully_connected_layer_set_kernel_weights(nvinfer1::IFullyConnectedLayer *layer,nvinfer1::Weights weights);
nvinfer1::Weights fully_connected_layer_get_kernel_weights(nvinfer1::IFullyConnectedLayer *layer);
void fully_connected_layer_set_bias_weights(nvinfer1::IFullyConnectedLayer *layer,nvinfer1::Weights weights);
nvinfer1::Weights fully_connected_layer_get_bias_weights(nvinfer1::IFullyConnectedLayer *layer);

#endif //LIBTRT_TRTFULLYCONNECTEDLAYER_H
