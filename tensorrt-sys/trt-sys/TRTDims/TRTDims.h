//
// Created by mason on 11/22/19.
//

#ifndef LIBTRT_TRTDIMS_H
#define LIBTRT_TRTDIMS_H

#include <NvInfer.h>
#include <NvInferLegacyDims.h>
#include "../TRTEnums.h"

nvinfer1::Dims create_dims(int nb_dims, const int* d, const DimensionType_t *dimension_types);
nvinfer1::Dims2 create_dims2(int dim1, int dim2);
nvinfer1::DimsHW create_dimsHW(int height, int width);
nvinfer1::Dims3 create_dims3(int dim1, int dim2, int dim3);
nvinfer1::Dims4 create_dims4(int dim1, int dim2, int dim3, int dim4);

#endif //LIBTRT_TRTDIMS_H
