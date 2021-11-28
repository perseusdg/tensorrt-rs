//
// Created by mason on 11/27/19.
//

#ifndef LIBTRT_TRTNETWORKDEFINITION_H
#define LIBTRT_TRTNETWORKDEFINITION_H

#include <stdint.h>
#include <NvInfer.h>

void destroy_network(nvinfer1::INetworkDefinition *network);
nvinfer1::ITensor *network_add_input(nvinfer1::INetworkDefinition *network, const char *name, nvinfer1::DataType dataType, nvinfer1::Dims dims);
nvinfer1::ITensor *network_get_input(nvinfer1::INetworkDefinition *network, int32_t idx);
int network_get_nb_layers(nvinfer1::INetworkDefinition *network);
nvinfer1::ILayer *network_get_layer(nvinfer1::INetworkDefinition *network, int index);
int network_get_nb_inputs(nvinfer1::INetworkDefinition *network);
int network_get_nb_outputs(nvinfer1::INetworkDefinition *network);
nvinfer1::ITensor *network_get_output(nvinfer1::INetworkDefinition *network, int32_t index);
void network_remove_tensor(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *tensor);
void network_mark_output(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *tensor);
void network_unmark_output(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *tensor);
nvinfer1::IIdentityLayer *network_add_identity_layer(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *inputTensor);
nvinfer1::IElementWiseLayer *network_add_element_wise(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input1, nvinfer1::ITensor *input2, nvinfer1::ElementWiseOperation op);
nvinfer1::IGatherLayer *network_add_gather(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *data, nvinfer1::ITensor *indices, int32_t axis);
nvinfer1::IActivationLayer *network_add_activation(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input, nvinfer1::ActivationType type);
nvinfer1::IPoolingLayer *network_add_pooling(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input, nvinfer1::PoolingType poolingType, nvinfer1::DimsHW dims);
nvinfer1::IConcatenationLayer *network_add_concatenation(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input,int32_t nbInputs);
nvinfer1::IResizeLayer *network_add_resize(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input);
nvinfer1::IReduceLayer *network_add_reduce(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input,nvinfer1::ReduceOperation reduceOperation,uint32_t axes,bool keepDimensions);
nvinfer1::IPaddingLayer *network_add_padding(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input,nvinfer1::Dims prePadding,nvinfer1::Dims postPadding);
nvinfer1::IScaleLayer *network_add_scale(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input,nvinfer1::ScaleMode scaleMode,nvinfer1::Weights shift,nvinfer1::Weights scale,nvinfer1::Weights power);
nvinfer1::IScatterLayer *network_add_scatter(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *data,nvinfer1::ITensor *indices,nvinfer1::ITensor *updates,nvinfer1::ScatterMode scatterMode);
nvinfer1::ISliceLayer *network_add_slice(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input,nvinfer1::Dims start,nvinfer1::Dims size,nvinfer1::Dims stride);
nvinfer1::IQuantizeLayer *network_add_quantize(nvinfer1::INetworkDefinition *network,nvinfer1::ITensor *input,nvinfer1::ITensor *scale);


#endif //LIBTRT_TRTNETWORKDEFINITION_H
