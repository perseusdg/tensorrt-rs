//
// Created by perseusdg on 11/18/21.
//

#ifndef LIBTRT_TRTRESIZELAYER_H
#define LIBTRT_TRTRESIZELAYER_H
#include "../TRTEnums.h"
#include "../TRTDims/TRTDims.h"
#include "TRTLayer.h"


void resize_layer_set_output_dimensions(nvinfer1::IResizeLayer* layer,nvinfer1::Dims dimensions);
nvinfer1::Dims resize_layer_get_output_dimensions(nvinfer1::IResizeLayer* layer);
void resize_layer_set_scales(nvinfer1::IResizeLayer* layer,const float* scales,int32_t nbScales);
int32_t resize_layer_get_scales(nvinfer1::IResizeLayer* layer,int32_t size,float* scales);
void resize_layer_set_resize_mode(nvinfer1::IResizeLayer* layer,ResizeMode_t resizeMode);
ResizeMode_t resize_layer_get_resize_mode(nvinfer1::IResizeLayer* layer);
void resize_layer_set_coordinate_transformation(nvinfer1::IResizeLayer* layer,ResizeCoordinateTransformation_t coordTransform);
ResizeCoordinateTransformation_t resize_layer_get_coordinate_transformation(nvinfer1::IResizeLayer* layer);
void resize_layer_set_selector_for_single_pixel(nvinfer1::IResizeLayer* layer,ResizeSelector_t selector);
ResizeSelector_t resize_layer_get_selector_for_single_pixel(nvinfer1::IResizeLayer* layer);
void resize_layer_set_nearest_rounding(nvinfer1::IResizeLayer* layer,ResizeRoundMode_t mode);
ResizeRoundMode_t resize_layer_get_nearest_rounding(nvinfer1::IResizeLayer* layer);
void resize_Layer_set_input(nvinfer1::IResizeLayer* layer,int32_t index,nvinfer1::ITensor *tensor);


#endif //LIBTRT_TRTRESIZELAYER_H
