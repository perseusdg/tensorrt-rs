//
// Created by perseusdg on 11/29/21.
//

#ifndef LIBTRT_TRTDECONVOLUTIONLAYER_H
#define LIBTRT_TRTDECONVOLUTIONLAYER_H

#include "../TRTEnums.h"
#include "TRTLayer.h"

void deconvolution_layer_set_nb_output_maps(nvinfer1::IDeconvolutionLayer *layer,int32_t nbOutputMaps);
int32_t deconvolution_layer_get_nb_output_maps(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_nb_groups(nvinfer1::IDeconvolutionLayer *layer,int32_t nbGroups);
int32_t deconvolution_layer_get_nb_groups(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_kernel_weights(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Weights weights);
nvinfer1::Weights deconvolution_layer_get_kernel_weights(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_bias_weights(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Weights weights);
nvinfer1::Weights deconvolution_layer_get_bias_weights(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_pre_padding(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims deconvolution_layer_get_pre_padding(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_post_padding(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims deconvolution_layer_get_post_padding(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_padding_mode(nvinfer1::IDeconvolutionLayer *layer,PaddingMode_t paddingMode);
PaddingMode_t deconvolution_layer_get_padding_mode(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_kernel_size_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims kernelSize);
nvinfer1::Dims deconvolution_layer_get_kernel_size_nd(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_stride_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims stride);
nvinfer1::Dims deconvolution_layer_get_stride_nd(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_padding_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims padding);
nvinfer1::Dims deconvolution_layer_get_padding_nd(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_dilation_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims dilation);
nvinfer1::Dims deconvolution_layer_get_dilation_nd(nvinfer1::IDeconvolutionLayer *layer);
void deconvolution_layer_set_input(nvinfer1::IDeconvolutionLayer *layer,int32_t index,nvinfer1::ITensor *input);


#endif //LIBTRT_TRTDECONVOLUTIONLAYER_H
