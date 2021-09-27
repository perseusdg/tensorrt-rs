//
// Created by mason on 11/27/19.
//

#ifndef LIBTRT_TRTBUILDER_H
#define LIBTRT_TRTBUILDER_H

#include <NvInfer.h>
#include "../TRTLogger/TRTLogger.h"
#include "../TRTEnums.h"

#include <stddef.h>
#include <stdint.h>

nvinfer1::ICudaEngine *build_cuda_engine(nvinfer1::IBuilder *builder, nvinfer1::INetworkDefinition *network);
nvinfer1::IBuilder *create_infer_builder(Logger_t *logger);
#if defined(TRT8)
void create_infer_builder_config(nvinfer1::IBuilder *builder,nvinfer1::IBuilderConfig* config);
void builder_config_set_max_workspace_size(nvinfer1::IBuilderConfig* config,size_t batch_size);
void builder_config_set_debug_flag(nvinfer1::IBuilderConfig* config);
void builder_config_set_fp16(nvinfer1::IBuilderConfig* config);
void builder_config_set_int8(nvinfer1::IBuilderConfig* config);
void builder_config_set_dla_core(nvinfer1::IBuilderConfig* config,int dla_core);
void builder_config_set_default_device_type_dla(nvinfer1::IBuilderConfig* config);
void builder_config_set_gpu_fallback(nvinfer1::IBuilderConfig* config);
void builder_config_set_avg_timing_iterations(nvinfer1::IBuilderConfig* config,int time);
void builder_config_set_min_timing_iterations(nvinfer1::IBuilderConfig* config,int time);
void builder_config_reset(nvinfer1::IBuilderConfig* config);
#endif 


#if defined(TRT7) || defined(TRT6)
void builder_set_max_workspace_size(nvinfer1::IBuilder* builder, size_t batch_size);
size_t builder_get_max_workspace_size(nvinfer1::IBuilder* builder);
void builder_set_half2_mode(nvinfer1::IBuilder* builder, bool mode);
bool builder_get_half2_mode(nvinfer1::IBuilder* builder);
void builder_set_debug_sync(nvinfer1::IBuilder* builder, bool sync);
bool builder_get_debug_sync(nvinfer1::IBuilder* builder);
void builder_set_min_find_iterations(nvinfer1::IBuilder* builder, int min_find);
int builder_get_min_find_iterations(nvinfer1::IBuilder* builder);
void builder_set_average_find_iterations(nvinfer1::IBuilder* builder, int avg_find);
int builder_get_average_find_iterations(nvinfer1::IBuilder* builder);
void builder_set_int8_mode(nvinfer1::IBuilder* builder, bool mode);
bool builder_get_int8_mode(nvinfer1::IBuilder* builder);
void builder_set_fp16_mode(nvinfer1::IBuilder* builder, bool mode);
bool builder_get_fp16_mode(nvinfer1::IBuilder* builder);
void builder_set_device_type(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer, DeviceType_t deviceType);
DeviceType_t  builder_get_device_type(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer);
bool builder_is_device_type_set(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer);
void builder_set_default_device_type(nvinfer1::IBuilder* builder, DeviceType_t deviceType);
DeviceType_t  builder_get_default_device_type(nvinfer1::IBuilder* builder);
void builder_reset_device_type(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer);
bool builder_can_run_on_dla(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer);
void builder_allow_gpu_fallback(nvinfer1::IBuilder* builder, bool set_fallback_mode);
void builder_set_refittable(nvinfer1::IBuilder* builder, bool can_refit);
bool builder_get_refittable(nvinfer1::IBuilder* builder);
void builder_set_dla_core(nvinfer1::IBuilder* builder, int dla_core);
int builder_get_dla_core(nvinfer1::IBuilder* builder);
void builder_set_strict_type_constraints(nvinfer1::IBuilder* builder, bool mode);
bool builder_get_strict_type_constraints(nvinfer1::IBuilder* builder);
#endif
void destroy_builder(nvinfer1::IBuilder* builder);
void builder_set_max_batch_size(nvinfer1::IBuilder* builder, int32_t batch_size);
int32_t builder_get_max_batch_size(nvinfer1::IBuilder* builder);
bool builder_platform_has_fast_fp16(nvinfer1::IBuilder* builder);
bool builder_platform_has_fast_int8(nvinfer1::IBuilder* builder);
int builder_get_max_dla_batch_size(nvinfer1::IBuilder* builder);
int builder_get_nb_dla_cores(nvinfer1::IBuilder* builder);

void builder_set_engine_capability(nvinfer1::IBuilder* builder, EngineCapabiliy_t engine_capability);
EngineCapabiliy_t builder_get_engine_capability(nvinfer1::IBuilder* builder);
#if defined(TRT6) || defined(TRT7) || defined(TRT8)
nvinfer1::INetworkDefinition *create_network_v2(nvinfer1::IBuilder* builder, uint32_t flags);
#else
nvinfer1::INetworkDefinition *create_network(nvinfer1::IBuilder* builder);
#endif

void builder_reset(nvinfer1::IBuilder* builder);
#endif //LIBTRT_TRTBUILDER_H
