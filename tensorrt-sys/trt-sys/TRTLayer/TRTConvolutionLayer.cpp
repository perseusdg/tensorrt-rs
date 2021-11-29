//
// Created by perseusdg on 11/29/21.
//

#include "TRTConvolutionLayer.h"

void convolution_layer_set_nb_output_maps(nvinfer1::IConvolutionLayer *layer,int32_t nbOutputMaps){
    layer->setNbOutputMaps(nbOutputMaps);
}

int32_t convolution_layer_get_nb_output_maps(nvinfer1::IConvolutionLayer *layer){
    return (layer->getNbOutputMaps());
}

void convolution_layer_set_nb_groups(nvinfer1::IConvolutionLayer *layer,int32_t nb_groups){
    layer->setNbGroups(nb_groups);
}

int32_t convolution_layer_get_nb_groups(nvinfer1::IConvolutionLayer *layer){
    return(layer->getNbGroups());
}

void convolution_layer_set_kernel_weights(nvinfer1::IConvolutionLayer *layer,nvinfer1::Weights weights){
    layer->setKernelWeights(weights);
}

nvinfer1::Weights convolution_layer_get_kernel_weights(nvinfer1::IConvolutionLayer *layer){
    return(layer->getKernelWeights());
}

void convolution_layer_set_bias_weights(nvinfer1::IConvolutionLayer *layer,nvinfer1::Weights bias){
    layer->setBiasWeights(bias);
}

nvinfer1::Weights convolution_layer_get_bias_weights(nvinfer1::IConvolutionLayer *layer){
    return(layer->getBiasWeights());
}

void convolution_layer_set_pre_padding(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims padding){
    layer->setPrePadding(padding);
}

nvinfer1::Dims convolution_layer_get_pre_padding(nvinfer1::IConvolutionLayer *layer){
    return(layer->getPrePadding());
}

void convolution_layer_set_post_padding(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims padding){
    layer->setPostPadding(padding);
}

nvinfer1::Dims convolution_layer_get_post_padding(nvinfer1::IConvolutionLayer *layer){
    return(layer->getPostPadding());
}

void convolution_layer_set_padding_mode(nvinfer1::IConvolutionLayer *layer,PaddingMode_t paddingMode){
    layer->setPaddingMode(static_cast<nvinfer1::PaddingMode>(paddingMode));
}

PaddingMode_t convolution_layer_get_padding_mode(nvinfer1::IConvolutionLayer *layer){
    return(static_cast<PaddingMode_t>(layer->getPaddingMode()));
}

void convolution_layer_set_kernel_size_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims kernelSize){
    layer->setKernelSizeNd(kernelSize);
}

nvinfer1::Dims convolution_layer_get_kernel_size_nd(nvinfer1::IConvolutionLayer *layer){
    return(layer->getKernelSizeNd());
}

void convolution_layer_set_stride_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims stride){
    layer->setStrideNd(stride);
}

nvinfer1::Dims convolution_layer_get_stride_nd(nvinfer1::IConvolutionLayer *layer){
    return(layer->getStrideNd());
}

void convolution_layer_set_padding_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims padding){
    layer->setPaddingNd(padding);
}

nvinfer1::Dims convolution_layer_get_padding_nd(nvinfer1::IConvolutionLayer *layer){
    return (layer->getPaddingNd());
}

void convolution_layer_set_dilation_nd(nvinfer1::IConvolutionLayer *layer,nvinfer1::Dims dilation){
    layer->setDilationNd(dilation);
}

nvinfer1::Dims convolution_layer_get_dilation_nd(nvinfer1::IConvolutionLayer *layer){
    return(layer->getDilationNd());
}

nvinfer1::Dims convolution_layer_set_input(nvinfer1::IConvolutionLayer *layer,int32_t index,nvinfer1::ITensor *input){
    layer->setInput(index,*input);
}

