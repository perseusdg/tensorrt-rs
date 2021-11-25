//
// Created by mason on 11/27/19.
//
#include <memory>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "TRTBuilder.h"
#include "../TRTLogger/TRTLoggerInternal.hpp"

void builder_set_max_batch_size(nvinfer1::IBuilder* builder, int32_t batch_size) {
    builder->setMaxBatchSize(batch_size);
}

int32_t builder_get_max_batch_size(nvinfer1::IBuilder* builder) {
    return builder->getMaxBatchSize();
}

int32_t builder_get_dla_max_batch_size(nvinfer1::IBuilder *builder){
    return builder->getMaxDLABatchSize();
}

int32_t builder_get_nb_dla_cores(nvinfer1::IBuilder *builder){
    return builder->getNbDLACores();
}

nvinfer1::ICudaEngine *build_cuda_engine_with_config(nvinfer1::IBuilder *builder,nvinfer1::IBuilderConfig* config,nvinfer1::INetworkDefinition *network){
    return builder->buildEngineWithConfig(*network,*config);
}


nvinfer1::IBuilderConfig* create_infer_builder_config(nvinfer1::IBuilder *builder){
    return builder->createBuilderConfig();
}
void builder_config_set_max_workspace_size(nvinfer1::IBuilderConfig* config,size_t batch_size){
    config->setMaxWorkspaceSize(batch_size);
}

void builder_config_set_dla_core(nvinfer1::IBuilderConfig* config,int dla_core){
    config->setDLACore(dla_core);
}

int builder_config_get_dla_core(nvinfer1::IBuilderConfig* config){
    return(config->getDLACore());
}

void builder_config_set_default_global_device_type(nvinfer1::IBuilderConfig* config,DeviceType_t deviceType){
    config->setDefaultDeviceType(static_cast<nvinfer1::DeviceType>(deviceType));
}

void builder_config_set_gpu_fallback(nvinfer1::IBuilderConfig* config){
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
}

void builder_config_set_avg_timing_iterations(nvinfer1::IBuilderConfig* config,int time){
    config->setAvgTimingIterations(time);
}

void builder_config_set_min_timing_iterations(nvinfer1::IBuilderConfig* config,int time){
    config->setMinTimingIterations(time);
}

int32_t builder_config_get_avg_timing_iterations(nvinfer1::IBuilderConfig *config){
    return config->getAvgTimingIterations();
}

int32_t builder_config_get_min_timing_iterations(nvinfer1::IBuilderConfig *config){
    return config->getMinTimingIterations();
}

void builder_config_set_int8_mode(nvinfer1::IBuilderConfig* config){
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
}

bool builder_config_get_int8_mode(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kINT8);
}

void builder_config_set_fp16_mode(nvinfer1::IBuilderConfig* config){
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
}

bool builder_config_get_fp16_mode(nvinfer1::IBuilderConfig* config){
    config->getFlag(nvinfer1::BuilderFlag::kFP16);
}

DeviceType_t builder_config_get_default_global_device_type(nvinfer1::IBuilderConfig *config){
    return static_cast<DeviceType_t>(config->getDefaultDeviceType());
}
void builder_config_set_device_type_layer(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer,DeviceType_t deviceType){
    config->setDeviceType(layer,static_cast<nvinfer1::DeviceType>(deviceType));
}
DeviceType_t builder_config_get_device_type_layer(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer){
    return static_cast<DeviceType_t>(config->getDeviceType(layer));
}

bool builder_config_is_device_type_set(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer){
    return(config->isDeviceTypeSet(layer));
}

void builder_config_reset_device_type(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer){
    config->resetDeviceType(layer);
}

bool builder_config_run_on_dla(nvinfer1::IBuilderConfig* config,nvinfer1::ILayer* layer){
    return(config->canRunOnDLA(layer));
}

void builder_config_set_strict_type_constraints(nvinfer1::IBuilderConfig* config,bool mode){
    if(mode) {
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }
}

bool builder_config_get_strict_type_constraints(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);

}

void builder_config_set_refittable_engine(nvinfer1::IBuilderConfig* config,bool can_refit){
    if(can_refit){
        config->setFlag(nvinfer1::BuilderFlag::kREFIT);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kREFIT);
    }
}

bool builder_config_get_refittable_engine(nvinfer1::IBuilderConfig*config){
    return config->getFlag(nvinfer1::BuilderFlag::kREFIT);
}

void builder_config_set_debug_sync(nvinfer1::IBuilderConfig* config,bool sync){
    if(sync){
        config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kDEBUG);
    }
}

bool builder_config_get_debug_sync(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kDEBUG);
}

void builder_config_set_sparse_weights(nvinfer1::IBuilderConfig* config,bool mode){
    if(mode){
        config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    }
}

bool builder_config_get_sparse_weights(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
}

void builder_config_set_disable_timing_cache(nvinfer1::IBuilderConfig* config,bool mode){
    if(mode){
        config->setFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
    }
}

bool builder_config_get_disable_timing_cache(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
}
void builder_config_set_tf32(nvinfer1::IBuilderConfig* config,bool mode){
    if(mode){
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    }
}

bool builder_config_get_tf32(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kTF32);
}

void builder_config_set_safety_scope(nvinfer1::IBuilderConfig* config,bool mode){
    if(mode){
        config->setFlag(nvinfer1::BuilderFlag::kSAFETY_SCOPE);
    }else{
        config->clearFlag(nvinfer1::BuilderFlag::kSAFETY_SCOPE);
    }
}

bool builder_config_get_safety_scope(nvinfer1::IBuilderConfig* config){
    return config->getFlag(nvinfer1::BuilderFlag::kSAFETY_SCOPE);
}


size_t builder_config_get_max_workspace_size(nvinfer1::IBuilderConfig* config){
    return config->getMaxWorkspaceSize();
}

void builder_config_set_engine_capability(nvinfer1::IBuilderConfig* config,EngineCapabiliy_t engineCapabiliy){
    config->setEngineCapability(static_cast<nvinfer1::EngineCapability>(engineCapabiliy));
}
EngineCapabiliy_t builder_config_get_engine_capability(nvinfer1::IBuilderConfig* config){
    return static_cast<EngineCapabiliy_t>(config->getEngineCapability());
}

void builder_config_set_profile_stream(nvinfer1::IBuilderConfig* config,cudaStream_t stream){
    config->setProfileStream(stream);
}

cudaStream_t builder_config_get_profile_stream(nvinfer1::IBuilderConfig* config){
    return config->getProfileStream();
}



void builder_config_reset(nvinfer1::IBuilderConfig* config){
    config->reset();
}

bool builder_platform_has_fast_fp16(nvinfer1::IBuilder* builder){
    return builder->platformHasFastFp16();
}

bool builder_platform_has_fast_int8(nvinfer1::IBuilder* builder) {
    return builder->platformHasFastInt8();
}


int builder_get_max_dla_batch_size(nvinfer1::IBuilder* builder) {
    return builder->getMaxBatchSize();
}

nvinfer1::IBuilder *create_infer_builder(Logger_t *logger) {
    initLibNvInferPlugins(&logger->getLogger(), "");
    return nvinfer1::createInferBuilder(logger->getLogger());
}


void destroy_builder(nvinfer1::IBuilder* builder) {
    builder->destroy();
}


nvinfer1::INetworkDefinition *create_network_v2(nvinfer1::IBuilder *builder, uint32_t flags) {
    return builder->createNetworkV2(flags);
}


void builder_reset(nvinfer1::IBuilder* builder) {
    builder->reset();
}
