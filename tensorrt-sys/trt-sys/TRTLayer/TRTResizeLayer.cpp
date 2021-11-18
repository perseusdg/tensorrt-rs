//
// Created by perseusdg on 11/18/21.
//

#include "TRTResizeLayer.h"

void resize_layer_set_output_dimensions(nvinfer1::IResizeLayer* layer,nvinfer1::Dims dimensions){
    layer->setOutputDimensions(dimensions);
}

nvinfer1::Dims resize_layer_get_output_dimensions(nvinfer1::IResizeLayer* layer){
    return (layer->getOutputDimensions());
}

void resize_layer_set_scales(nvinfer1::IResizeLayer* layer,const float* scales,int32_t nbScales){
    layer->setScales(scales,nbScales);
}

int32_t resize_layer_get_scales(nvinfer1::IResizeLayer* layer,int32_t size,float* scales){
    return (layer->getScales(size,scales));
}

void resize_layer_set_resize_mode(nvinfer1::IResizeLayer* layer,ResizeMode_t resizeMode){
    layer->setResizeMode(static_cast<nvinfer1::ResizeMode>(resizeMode));
}

ResizeMode_t resize_layer_get_resize_mode(nvinfer1::IResizeLayer* layer){
    return (static_cast<ResizeMode_t>(layer->getResizeMode()));
}

void resize_layer_set_coordinate_transformation(nvinfer1::IResizeLayer* layer,ResizeCoordinateTransformation_t coordTransform){
    layer->setCoordinateTransformation(static_cast<nvinfer1::ResizeCoordinateTransformation>(coordTransform));
}

ResizeCoordinateTransformation_t resize_layer_get_coordinate_transformation(nvinfer1::IResizeLayer* layer){
    return ((static_cast<ResizeCoordinateTransformation_t>(layer->getCoordinateTransformation())));
}

void resize_layer_set_selector_for_single_pixel(nvinfer1::IResizeLayer* layer,ResizeSelector_t selector){
    layer->setSelectorForSinglePixel(static_cast<nvinfer1::ResizeSelector>(selector));
}

ResizeSelector_t resize_layer_get_selector_for_single_pixel(nvinfer1::IResizeLayer* layer){
    return (static_cast<ResizeSelector_t>(layer->getSelectorForSinglePixel()));
}

void resize_layer_set_nearest_rounding(nvinfer1::IResizeLayer* layer,ResizeRoundMode_t mode){
    layer->setNearestRounding(static_cast<nvinfer1::ResizeRoundMode>(mode));
}

ResizeRoundMode_t resize_layer_get_nearest_rounding(nvinfer1::IResizeLayer* layer){
    return (static_cast<ResizeRoundMode_t>(layer->getNearestRounding()));
}

void resize_Layer_set_input(nvinfer1::IResizeLayer* layer,int32_t index,nvinfer1::ITensor *tensor){
    layer->setInput(index,*tensor);
}
