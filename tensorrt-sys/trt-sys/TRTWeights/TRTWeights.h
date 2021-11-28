//
// Created by perseusdg on 28/11/21.
//

#ifndef LIBTRT_TRTWEIGHTS_H
#define LIBTRT_TRTWEIGHTS_H

#include <NvInfer.h>
#include "../TRTEnums.h"

nvinfer1::Weights create_weights(DataType_t dataType,const void* values,int64_t count);



#endif //LIBTRT_TRTWEIGHTS_H
