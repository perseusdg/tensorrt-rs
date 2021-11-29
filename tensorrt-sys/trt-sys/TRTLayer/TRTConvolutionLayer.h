//
// Created by perseusdg on 11/29/21.
//

#ifndef LIBTRT_TRTCONVOLUTIONLAYER_H
#define LIBTRT_TRTCONVOLUTIONLAYER_H

#include "TRTLayer.h"
#include "../TRTEnums.h"

void convolution_layer_set_nb_output_maps(nvinfer1::IConvolutionLayer *layer,int32_t nbOutputMaps);
int32_t convolution_layer_get_nb_output_maps(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_nb_groups(nvinfer1::IConvolutionLayer *layer,int32_t nb_groups);
int32_t convolution_layer_get_nb_groups(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_kernel_weights(nvinfer1::IConvolutionLayer *layer,nvinfer1::Weights weights);
nvinfer1::Weights convolution_layer_get_kernel_weights(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_bias_weights(nvinfer1::IConvolutionLayer *layer,nvinfer1::Weights bias);
nvinfer1::Weights convolution_layer_get_bias_weights(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_pre_padding(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims convolution_layer_get_pre_padding(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_post_padding(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims convolution_layer_get_post_padding(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_padding_mode(nvinfer1::IConvolutionLayer *layer,PaddingMode_t paddingMode);
PaddingMode_t convolution_layer_get_padding_mode(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_kernel_size_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims kernelSize);
nvinfer1::Dims convolution_layer_get_kernel_size_nd(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_stride_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims stride);
nvinfer1::Dims convolution_layer_get_stride_nd(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_padding_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims convolution_layer_get_padding_nd(nvinfer1::IConvolutionLayer *layer);
void convolution_layer_set_dilation_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims dilation);
nvinfer1::Dims convolution_layer_get_dilation_nd(nvinfer1::IConvolutionLayer *layer);
nvinfer1::Dims convolution_layer_set_input(nvinfer1::IConvolutionLayer *layer,int32_t index,nvinfer1::ITensor *input);
#endif //LIBTRT_TRTCONVOLUTIONLAYER_H
