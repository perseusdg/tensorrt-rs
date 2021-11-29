//
// Created by perseusdg on 11/29/21.
//

#include "TRTDeconvolutionLayer.h"

void deconvolution_layer_set_nb_output_maps(nvinfer1::IDeconvolutionLayer *layer,int32_t nbOutputMaps){
    layer->setNbOutputMaps(nbOutputMaps);
}

int32_t deconvolution_layer_get_nb_output_maps(nvinfer1::IDeconvolutionLayer *layer){
    return (layer->getNbOutputMaps());
}

void deconvolution_layer_set_nb_groups(nvinfer1::IDeconvolutionLayer *layer,int32_t nbGroups){
    layer->setNbGroups(nbGroups);
}

int32_t deconvolution_layer_get_nb_groups(nvinfer1::IDeconvolutionLayer *layer){
    return (layer->getNbGroups());
}

void deconvolution_layer_set_kernel_weights(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Weights weights){
    layer->setKernelWeights(weights);
}

nvinfer1::Weights deconvolution_layer_get_kernel_weights(nvinfer1::IDeconvolutionLayer *layer){
    return (layer->getKernelWeights());
}

void deconvolution_layer_set_bias_weights(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Weights weights){
    layer->setBiasWeights(weights);
}

nvinfer1::Weights deconvolution_layer_get_bias_weights(nvinfer1::IDeconvolutionLayer *layer){
    return (layer->getBiasWeights());
}

void deconvolution_layer_set_pre_padding(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims padding){
    layer->setPrePadding(padding);
}

nvinfer1::Dims deconvolution_layer_get_pre_padding(nvinfer1::IDeconvolutionLayer *layer){
    return(layer->getPrePadding());
}

void deconvolution_layer_set_post_padding(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims padding){
    layer->setPostPadding(padding);
}

nvinfer1::Dims deconvolution_layer_get_post_padding(nvinfer1::IDeconvolutionLayer *layer){
    return(layer->getPostPadding());
}

void deconvolution_layer_set_padding_mode(nvinfer1::IDeconvolutionLayer *layer,PaddingMode_t paddingMode){
    layer->setPaddingMode(static_cast<nvinfer1::PaddingMode>(paddingMode));
}

PaddingMode_t deconvolution_layer_get_padding_mode(nvinfer1::IDeconvolutionLayer *layer){
    return(static_cast<PaddingMode_t>(layer->getPaddingMode()));
}

void deconvolution_layer_set_kernel_size_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims kernelSize){
    layer->setKernelSizeNd(kernelSize);
}

nvinfer1::Dims deconvolution_layer_get_kernel_size_nd(nvinfer1::IDeconvolutionLayer *layer){
    return(layer->getKernelSizeNd());
}

void deconvolution_layer_set_stride_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims stride){
    layer->setStrideNd(stride);
}

nvinfer1::Dims deconvolution_layer_get_stride_nd(nvinfer1::IDeconvolutionLayer *layer){
    return(layer->getStrideNd());
}

void deconvolution_layer_set_padding_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims padding){
    layer->setPaddingNd(padding);
}

nvinfer1::Dims deconvolution_layer_get_padding_nd(nvinfer1::IDeconvolutionLayer *layer){
    return (layer->getPaddingNd());
}

void deconvolution_layer_set_dilation_nd(nvinfer1::IDeconvolutionLayer *layer,nvinfer1::Dims dilation){
    layer->setDilationNd(dilation);
}

nvinfer1::Dims deconvolution_layer_get_dilation_nd(nvinfer1::IDeconvolutionLayer *layer){
    return(layer->getDilationNd());
}

void deconvolution_layer_set_input(nvinfer1::IDeconvolutionLayer *layer,int32_t index,nvinfer1::ITensor *input){
    layer->setInput(index,*input);
}