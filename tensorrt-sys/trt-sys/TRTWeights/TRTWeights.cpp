//
// Created by perseusdg on 28/11/21.
//

#include "TRTWeights.h"

nvinfer1::Weights create_weights(DataType_t dataType,const void* values,int64_t count){
    nvinfer1::Weights weights{};
    weights.type = static_cast<nvinfer1::DataType>(dataType);
    weights.values = values;
    weights.count = count;
    return weights;
}