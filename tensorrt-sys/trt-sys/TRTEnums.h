//
// Created by mason on 10/10/20.
//

#ifndef LIBTRT_TRTENUMS_H
#define LIBTRT_TRTENUMS_H

#ifdef __cplusplus
extern "C" {
#endif

enum class ActivationType {
    kRELU = 0,
    kSIGMOID = 1,
    kTANH = 2,
    kLEAKY_RELU = 3,
    kELU = 4,
    kSELU = 5,
    kSOFTSIGN = 6,
    kSOFTPLUS = 7,
    kCLIP = 8,
    kHARD_SIGMOID = 9,
    kSCALED_TANH = 10,
    kTHRESHOLDED_RELU = 11,
};
typedef enum ActivationType ActivationType_t;

enum class DataType {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
};
typedef enum DataType DataType_t;

enum class DeviceType {
    kGPU = 0,
    kDLA = 1,
};
typedef enum DeviceType DeviceType_t;

enum class DimensionType {
    kSPATIAL = 0,
    kCHANNEL = 1,
    kINDEX = 2,
    kSEQUENCE = 3
};
typedef enum DimensionType DimensionType_t;

enum class ElementWiseOperation {
    kSUM = 0,
    kPROD = 1,
    kMAX = 2,
    kMIN = 3,
    kSUB = 4,
    kDIV = 5,
    kPOW = 6,
};
typedef enum ElementWiseOperation ElementWiseOperation_t;

enum class EngineCapabiliy {
    kDEFAULT = 0,
    kSAFE_GPU = 1,
    kSAFE_DLA = 2,
};
typedef enum EngineCapabiliy EngineCapabiliy_t;

enum class LayerType {
    kCONVOLUTION = 0,
    kFULLY_CONNECTED = 1,
    kACTIVATION  = 2,
    kPOOLING = 3,
    kLRN = 4,
    kSCALE = 5,
    kSOFTMAX = 6,
    kDECONVOLUTION = 7,
    kCONCATENATION = 8,
    kELEMENTWISE = 9,
    kPLUGIN = 10,
    kRNN = 11,
    kUNARY = 12,
    kPADDING = 13,
    kSHUFFLE = 14,
    kREDUCE = 15,
    kTOPK = 16,
    kGATHER = 17,
    kMATRIX_MULTIPLY = 18,
    kRAGGED_SOFTMAX = 19,
    kCONSTANT = 20,
    kRNN_V2 = 21,
    kIDENTITY = 22,
    kPLUGIN_V2 = 23,
    kSLICE = 24,
};
typedef enum LayerType LayerType_t;

enum class PaddingMode {
    kEXPLICIT_ROUND_DOWN = 0,
    kEXPLICIT_ROUND_UP = 1,
    kSAME_UPPER = 2,
    kSAME_LOWER = 3,
    kCAFFE_ROUND_DOWN = 4,
    kCAFFE_ROUND_UP = 5,
};

typedef enum PaddingMode PaddingMode_t;

enum class PoolingType {
    kMAX = 0,
    kAVERAGE = 1,
    kMAX_AVERAGE_BLEND = 2,
};

enum class TensorLocation {
    kDEVICE = 0,
    kHOST = 1,
};
typedef enum TensorLocation TensorLocation_t;

enum class ReduceOperation{
    kSUM = 0,
    kPROD = 1,
    kMAX = 2,
    kMIN = 3,
    kAVG = 4,
};
typedef enum ReduceOperation ReduceOperation_t;

enum class ResizeMode{
    kNEAREST = 0,
    kLINEAR = 1,
};
typedef enum ResizeMode ResizeMode_t;

enum class ResizeCoordinateTransformation {
    kALIGN_CORNERS = 0,
    kASYMMETRIC = 1,
    kHALF_PIXEL = 2,
};
typedef enum ResizeCoordinateTransformation ResizeCoordinateTransformation_t;

enum class ResizeSelector {
    kFORMULA = 0,
    kUPPER = 1,
};
typedef enum ResizeSelector ResizeSelector_t;

enum class ResizeRoundMode {
    kHALF_UP = 0,
    KHALF_DOWN = 1,
    KFLOOR = 2,
    kCEIL = 3,
};
typedef enum ResizeRoundMode ResizeRoundMode_t;

enum class ScaleMode {
    kUNIFORM = 0,
    kCHANNEL = 1,
    kELEMENTWISE = 2,
};
typedef enum ScaleMode ScaleMode_t;

enum class ScatterMode {
    kELEMENT = 0,
    kND = 1,
};

typedef enum ScatterMode ScatterMode_t;

enum class SliceMode {
    kDEFAULT = 0,
    kWRAP = 1,
    kCLAMP = 2,
    kFILL = 3,
    kREFLECT = 4,
};

typedef enum SliceMode SliceMode_t;

enum class FillOperation{
    LINSPACE = 0,
    RANDOM_UNIFORM = 1,
};

typedef enum FillOperation FillOperation_t;

#ifdef __cplusplus
};
#endif

#endif //LIBTRT_TRTENUMS_H
