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
nvinfer1::ICudaEngine *build_cuda_engine_with_config(nvinfer1::IBuilder *builder,nvinfer1::IBuilderConfig* config,nvinfer1::INetworkDefinition *network);
nvinfer1::IBuilder *create_infer_builder(Logger_t *logger);
nvinfer1::IBuilderConfig* create_infer_builder_config(nvinfer1::IBuilder *builder);
void builder_config_set_max_workspace_size(nvinfer1::IBuilderConfig* config,size_t batch_size);
void builder_config_set_debug_flag(nvinfer1::IBuilderConfig* config);
void builder_config_set_fp16(nvinfer1::IBuilderConfig* config);
void builder_config_set_int8(nvinfer1::IBuilderConfig* config);
void builder_config_set_dla_core(nvinfer1::IBuilderConfig* config,int dla_core);
int builder_config_get_dla_core(nvinfer1::IBuilderConfig* config);
void builder_config_set_default_global_device_type(nvinfer1::IBuilderConfig* config,DeviceType_t deviceType);
void builder_config_set_gpu_fallback(nvinfer1::IBuilderConfig* config);
void builder_config_set_avg_timing_iterations(nvinfer1::IBuilderConfig* config,int time);
void builder_config_set_min_timing_iterations(nvinfer1::IBuilderConfig* config,int time);
void builder_config_set_int8_mode(nvinfer1::IBuilderConfig* config);
bool builder_config_get_int8_mode(nvinfer1::IBuilderConfig* config);
void builder_config_set_fp16_mode(nvinfer1::IBuilderConfig* config);
bool builder_config_get_fp16_mode(nvinfer1::IBuilderConfig* config);
DeviceType_t builder_config_get_default_global_device_type(nvinfer1::IBuilderConfig *config);
void builder_config_set_device_type_layer(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer,DeviceType_t deviceType);
DeviceType_t builder_config_get_device_type_layer(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer);
bool builder_config_is_device_type_set(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer);
void builder_config_reset_device_type(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer);
bool builder_config_run_on_dla(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer);
void builder_config_set_strict_type_constraints(nvinfer1::IBuilderConfig* config,bool mode);
bool builder_config_get_strict_type_constraints(nvinfer1::IBuilderConfig* config);
void builder_config_set_refittable_engine(nvinfer1::IBuilderConfig* config,bool can_refit);
bool builder_config_get_refittable_engine(nvinfer1::IBuilderConfig*config);
void builder_config_set_debug_sync(nvinfer1::IBuilderConfig* config,bool sync);
bool builder_config_get_debug_sync(nvinfer1::IBuilderConfig* config);
void builder_config_set_sparse_weights(nvinfer1::IBuilderConfig* config,bool mode);
bool builder_config_get_sparse_weights(nvinfer1::IBuilderConfig* config);
void builder_config_set_disable_timing_cache(nvinfer1::IBuilderConfig* config,bool mode);
bool builder_config_get_disable_timing_cache(nvinfer1::IBuilderConfig* config);
void builder_config_set_tf32(nvinfer1::IBuilderConfig* config,bool mode);
bool builder_config_get_tf32(nvinfer1::IBuilderConfig* config);
void builder_config_set_safety_scope(nvinfer1::IBuilderConfig* config,bool mode);
bool builder_config_get_safety_scope(nvinfer1::IBuilderConfig* config);
void builder_config_set_max_workspace_size(nvinfer1::IBuilderConfig* config,int32_t workspace_size);
size_t builder_config_get_max_workspace_size(nvinfer1::IBuilderConfig* config);
void builder_config_set_engine_capability(nvinfer1::IBuilderConfig* config,EngineCapabiliy_t engineCapabiliy);
EngineCapabiliy_t builder_config_get_engine_capability(nvinfer1::IBuilderConfig* config);
void builder_config_set_profile_stream(nvinfer1::IBuilderConfig* config,cudaStream_t stream);
cudaStream_t builder_config_get_profile_stream(nvinfer1::IBuilderConfig* config);
void builder_config_reset(nvinfer1::IBuilderConfig* config);
void destroy_builder(nvinfer1::IBuilder* builder);
void builder_set_max_batch_size(nvinfer1::IBuilder* builder, int32_t batch_size);
int32_t builder_get_max_batch_size(nvinfer1::IBuilder* builder);
bool builder_platform_has_fast_fp16(nvinfer1::IBuilder* builder);
bool builder_platform_has_fast_int8(nvinfer1::IBuilder* builder);
int builder_get_max_dla_batch_size(nvinfer1::IBuilder* builder);
int builder_get_nb_dla_cores(nvinfer1::IBuilder* builder);
void builder_set_engine_capability(nvinfer1::IBuilder* builder, EngineCapabiliy_t engine_capability);
EngineCapabiliy_t builder_get_engine_capability(nvinfer1::IBuilder* builder);
nvinfer1::INetworkDefinition *create_network_v2(nvinfer1::IBuilder* builder, uint32_t flags);
void builder_reset(nvinfer1::IBuilder* builder);
#endif //LIBTRT_TRTBUILDER_H
