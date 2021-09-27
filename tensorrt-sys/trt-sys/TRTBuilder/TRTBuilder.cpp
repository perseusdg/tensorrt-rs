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

#if defined(TRT7) || defined(TRT6)
void builder_set_max_workspace_size(nvinfer1::IBuilder* builder, size_t workspace_size) {
    builder->setMaxWorkspaceSize(workspace_size);
}

size_t builder_get_max_workspace_size(nvinfer1::IBuilder* builder) {
    return builder->getMaxWorkspaceSize();
}

void builder_set_half2_mode(nvinfer1::IBuilder* builder, bool mode) {
    builder->setHalf2Mode(mode);
}

bool builder_get_half2_mode(nvinfer1::IBuilder* builder) {
    return builder->getHalf2Mode();
}

void builder_set_debug_sync(nvinfer1::IBuilder* builder, bool sync) {
    builder->setDebugSync(sync);
}

bool builder_get_debug_sync(nvinfer1::IBuilder* builder) {
    return builder->getDebugSync();
}

void builder_set_min_find_iterations(nvinfer1::IBuilder* builder, int min_find) {
    builder->setMinFindIterations(min_find);
}

int builder_get_min_find_iterations(nvinfer1::IBuilder* builder) {
    return builder->getMinFindIterations();
}

void builder_set_average_find_iterations(nvinfer1::IBuilder* builder, int avg_find) {
    builder->setAverageFindIterations(avg_find);
}

int builder_get_average_find_iterations(nvinfer1::IBuilder* builder) {
    return builder->getAverageFindIterations();
}

void builder_set_int8_mode(nvinfer1::IBuilder* builder, bool mode) {
    builder->setInt8Mode(mode);
}

bool builder_get_int8_mode(nvinfer1::IBuilder* builder) {
    return builder->getInt8Mode();
}

void builder_set_fp16_mode(nvinfer1::IBuilder* builder, bool mode) {
    builder->setFp16Mode(mode);
}

bool builder_get_fp16_mode(nvinfer1::IBuilder* builder) {
    return builder->getFp16Mode();
}
void builder_set_device_type(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer, DeviceType_t deviceType) {
    builder->setDeviceType(layer, static_cast<nvinfer1::DeviceType>(deviceType));
}

DeviceType_t builder_get_device_type(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer) {
    return static_cast<DeviceType_t>(builder->getDeviceType(layer));
}

bool builder_is_device_type_set(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer) {
    return builder->isDeviceTypeSet(layer);
}

void builder_set_default_device_type(nvinfer1::IBuilder* builder, DeviceType_t deviceType) {
    builder->setDefaultDeviceType(static_cast<nvinfer1::DeviceType>(deviceType));
}

DeviceType_t builder_get_default_device_type(nvinfer1::IBuilder *builder) {
    return static_cast<DeviceType_t>(builder->getDefaultDeviceType());
}

void builder_reset_device_type(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer) {
   builder->resetDeviceType(layer);
}

bool builder_can_run_on_dla(nvinfer1::IBuilder* builder, nvinfer1::ILayer* layer) {
    return builder->canRunOnDLA(layer);
}

void builder_allow_gpu_fallback(nvinfer1::IBuilder* builder, bool set_fallback_mode) {
    builder->allowGPUFallback(set_fallback_mode);
}

void builder_set_refittable(nvinfer1::IBuilder* builder, bool can_refit) {
    builder->setRefittable(can_refit);
}

bool builder_get_refittable(nvinfer1::IBuilder* builder) {
    return builder->getRefittable();
}

void builder_set_engine_capability(nvinfer1::IBuilder* builder, EngineCapabiliy_t engine_capability) {
    builder->setEngineCapability(static_cast<nvinfer1::EngineCapability>(engine_capability));
}

void builder_set_dla_core(nvinfer1::IBuilder* builder, int dla_core) {
    builder->setDLACore(dla_core);
}

int builder_get_dla_core(nvinfer1::IBuilder* builder) {
    return builder->getDLACore();
}

void builder_set_strict_type_constraints(nvinfer1::IBuilder* builder, bool mode) {
    builder->setStrictTypeConstraints(mode);
}

bool builder_get_strict_type_constraints(nvinfer1::IBuilder* builder) {
    return builder->getStrictTypeConstraints();
}

EngineCapabiliy_t builder_get_engine_capability(nvinfer1::IBuilder* builder) {
    return static_cast<EngineCapabiliy_t>(builder->getEngineCapability());
}

nvinfer1::ICudaEngine *build_cuda_engine(nvinfer1::IBuilder *builder, nvinfer1::INetworkDefinition *network) {
    return builder->buildCudaEngine(*network);
}

#endif 

#ifdef defined(TRT8)

void create_infer_builder_config(nvinfer::IBuilder *builder,nvinfer::IBuilderConfig* config){
    config = builder->createBuilderConfig();
}
void builder_config_set_max_workspace_size(nvinfer1::IBuilderConfig* config,size_t batch_size){
    config->setMaxWorkspaceSize(batch_size);
}

void builder_config_set_debug_flag(nvinfer1::IBuilderConfig* config){
    config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
}

void builder_config_set_fp16(nvinfer1::IBuilderConfig* config){
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
}

void builder_config_set_int8(nvinfer1::IBuilderConfig* config){
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
}

void builder_config_set_dla_core(nvinfer1::IBuilderConfig* config,int dla_core){
    config->setDLACore(dla_core);
}

void builder_config_set_default_device_type_dla(nvinfer1::IBuilderConfig* config){
    config->setDefaultDeviceType(nvinfer1::BuilderFlag::kDLA);
}

void builder_config_set_gpu_fallback(){
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
}

void builder_config_set_avg_timing_iterations(nvinfer1::IBuilderConfig* config,int time){
    config->setAvgTimingIterations(time);
}

void builder_config_set_min_timing_iterations(nvinfer1::IBuilderConfig* config,int time){
    config->setMinTimingIterations(time);
}

void builder_config_reset(nvinfer1::IBuilderConfig* config){
    config->reset();
}

#endif


bool builder_platform_has_fast_fp16(nvinfer1::IBuilder* builder){
    return builder->platformHasFastFp16();
}

bool builder_platform_has_fast_int8(nvinfer1::IBuilder* builder) {
    return builder->platformHasFastInt8();
}


int builder_get_max_dla_batch_size(nvinfer1::IBuilder* builder) {
    return builder->getMaxBatchSize();
}


int builder_get_nb_dla_cores(nvinfer1::IBuilder* builder) {
    return builder->getNbDLACores();
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
