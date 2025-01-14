#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use crate::engine::Engine;
use crate::network::layer::Layer;
use crate::network::Network;
use crate::runtime::Logger;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use std::os::raw::c_int;
#[cfg(feature = "trt-5")]
use tensorrt_sys::create_network;
#[cfg(not(feature = "trt-5"))]
use tensorrt_sys::create_network_v2;

use tensorrt_sys::{
    build_cuda_engine_with_config,create_infer_builder,create_infer_builder_config,
    builder_config_set_max_workspace_size,builder_config_set_dla_core,builder_config_get_dla_core,
    builder_config_set_default_global_device_type,builder_config_set_gpu_fallback,builder_get_dla_max_batch_size,builder_get_nb_dla_cores,
    builder_config_set_avg_timing_iterations,builder_config_set_min_timing_iterations,builder_config_set_int8_mode,
    builder_config_get_min_timing_iterations,builder_config_get_avg_timing_iterations,
    builder_config_set_fp16_mode,builder_config_get_int8_mode,builder_config_get_fp16_mode,builder_config_get_default_global_device_type,
    builder_config_set_device_type_layer,builder_config_get_device_type_layer,builder_config_is_device_type_set,
    builder_config_reset_device_type,builder_config_run_on_dla,builder_config_set_strict_type_constraints,
    builder_config_get_strict_type_constraints,builder_config_set_refittable_engine,builder_config_get_refittable_engine,
    builder_config_set_debug_sync,builder_config_get_debug_sync,builder_config_set_sparse_weights,builder_config_get_sparse_weights,
    builder_config_set_disable_timing_cache,builder_config_get_disable_timing_cache,builder_config_set_tf32,builder_config_get_tf32,
    builder_config_set_safety_scope,builder_config_get_safety_scope,builder_config_get_max_workspace_size,
    builder_set_engine_capability,builder_get_engine_capability,builder_config_set_profile_stream,builder_config_get_profile_stream,
    builder_config_reset,destroy_builder,builder_set_max_batch_size,builder_platform_has_fast_fp16,builder_platform_has_fast_int8,
    builder_get_max_batch_size,builder_get_max_dla_batch_size,builder_reset, builder_config_set_engine_capability,builder_config_get_engine_capability
};

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum DeviceType {
    GPU,
    DLA,
}

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum EngineCapability {
    Default,
    SafeGpu,
    SafeDla,
}

pub struct Builder<'a> {
    pub(crate) internal_builder: *mut tensorrt_sys::nvinfer1_IBuilder,
    pub(crate) internal_builder_config: *mut tensorrt_sys::nvinfer1_IBuilderConfig,
    pub(crate) logger: PhantomData<&'a Logger>,
}

bitflags! {
    pub struct NetworkBuildFlags: u32 {
        const DEFAULT = 0b0;
        const EXPLICIT_BATCH = 0b1;
        const EXPLICIT_PRECISION = 0b10;
    }
}

impl<'a> Builder<'a> {
    pub fn new(logger: &'a Logger) -> Self {
        let internal_builder = unsafe { create_infer_builder(logger.internal_logger) };
        let internal_builder_config = unsafe{create_infer_builder_config(internal_builder)};
        let logger = PhantomData;
        Self {
            internal_builder,
            internal_builder_config,
            logger,
        }
    }

    pub fn get_max_dla_batch_size(&self) -> i32 {
        unsafe {builder_get_dla_max_batch_size(self.internal_builder) as i32}
    }

    pub fn get_max_workspace_size(&self) -> usize {
        unsafe {builder_config_get_max_workspace_size(self.internal_builder_config) as usize}
    }

    pub fn get_nb_dla_cores(&self) -> i32 {
        unsafe {builder_get_nb_dla_cores(self.internal_builder) as i32}
    }

    pub fn set_max_workspace_size(&self, ws: usize) {
        unsafe {builder_config_set_max_workspace_size(self.internal_builder_config,ws as usize)}
    }

    pub fn get_max_batch_size(&self) -> i32 {
        unsafe { builder_get_max_batch_size(self.internal_builder) as i32 }
    }

    pub fn set_max_batch_size(&self, bs: i32) {
        unsafe { builder_set_max_batch_size(self.internal_builder, bs as i32) }
    }

    pub fn platform_has_fast_fp16(&self) -> bool {
        unsafe { builder_platform_has_fast_fp16(self.internal_builder) }
    }

    pub fn platform_has_fast_int8(&self) -> bool {
        unsafe { builder_platform_has_fast_int8(self.internal_builder) }
    }

    pub fn set_dla_core(&self,dla_core: i32){
        unsafe {builder_config_set_dla_core(self.internal_builder_config,dla_core)}
    }

    pub fn get_dla_core(&self) -> i32{
        unsafe {builder_config_get_dla_core(self.internal_builder_config)}
    }

    pub fn set_default_global_device_type(&self,device_type: DeviceType){
        unsafe {builder_config_set_default_global_device_type(self.internal_builder_config,device_type as c_int)}
    }

    pub fn set_gpu_fallback(&self){
        unsafe {builder_config_set_gpu_fallback(self.internal_builder_config)}
    }

    pub fn set_avg_timing_iterations(&self,time:i32){
        unsafe {builder_config_set_avg_timing_iterations(self.internal_builder_config,time)}
    }

    pub fn set_min_timing_iterations(&self,time:i32){
        unsafe{builder_config_set_min_timing_iterations(self.internal_builder_config,time)}
    }

    pub fn get_avg_timing_iterations(&self) -> i32 {
        unsafe {builder_config_get_avg_timing_iterations(self.internal_builder_config)}
    }

    pub fn get_min_timing_iterations(&self) -> i32 {
        unsafe {builder_config_get_min_timing_iterations(self.internal_builder_config)}
    }

    pub fn set_int8_mode(&self,mode:bool){
        if mode {
            unsafe { builder_config_set_int8_mode(self.internal_builder_config) }
        }
    }

    pub fn get_int8_mode(&self) -> bool{
        unsafe {builder_config_get_int8_mode(self.internal_builder_config)}
    }

    pub fn set_fp16_mode(&self, mode: bool) {
        unsafe { builder_config_set_fp16_mode(self.internal_builder_config)}
    }

    pub fn get_fp16_mode(&self) -> bool {
        unsafe {builder_config_get_fp16_mode(self.internal_builder_config)}
    }

    pub fn get_default_global_device_type(&self) -> DeviceType{
        let primitive = unsafe {builder_config_get_default_global_device_type(self.internal_builder_config)};
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_device_type<T: Layer>(&self, layer:&T, device_type:DeviceType){
        unsafe{builder_config_set_device_type_layer(self.internal_builder_config,layer.get_internal_layer(),device_type as c_int)}
    }

    pub fn get_device_type(&self, layer: &dyn Layer) -> DeviceType {
        let primitive =
            unsafe { builder_config_get_device_type_layer(self.internal_builder_config,layer.get_internal_layer()) };
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn is_device_type_set(&self, layer: &dyn Layer) -> bool {
        unsafe{builder_config_is_device_type_set(self.internal_builder_config,layer.get_internal_layer())}
    }
    pub fn reset_device_type(&self, layer: &dyn Layer) {
        unsafe { builder_config_reset_device_type(self.internal_builder_config,layer.get_internal_layer())}
    }

    pub fn can_run_on_dla(&self, layer: &dyn Layer) -> bool {
        unsafe{ builder_config_run_on_dla(self.internal_builder_config,layer.get_internal_layer())}
    }

    pub fn set_strict_type_constraints(&self,mode:bool) {
        unsafe{builder_config_set_strict_type_constraints(self.internal_builder_config,mode)}
    }

    pub fn get_strict_type_constraints(&self) -> bool{
        unsafe { builder_config_get_strict_type_constraints(self.internal_builder_config)}
    }

    pub fn set_refittable(&self, can_refit: bool) {
        unsafe { builder_config_set_refittable_engine(self.internal_builder_config,can_refit) }
    }

    pub fn get_refittable(&self) -> bool {
        unsafe { builder_config_get_refittable_engine(self.internal_builder_config) }
    }


    pub fn create_network_v2(&self, flags: NetworkBuildFlags) -> Network {
        let internal_network = unsafe { create_network_v2(self.internal_builder, flags.bits()) };
        Network { internal_network }
    }

    pub fn build_cuda_engine_with_config(&self, network: &Network) -> Engine{
        let internal_engine = unsafe{build_cuda_engine_with_config(self.internal_builder,self.internal_builder_config,network.internal_network)};
        Engine {internal_engine}
    }

    pub fn set_debug_sync(&self, mode:bool) {
        unsafe {builder_config_set_debug_sync(self.internal_builder_config,mode)}
    }

    pub fn get_debug_sync(&self) -> bool {
        unsafe {builder_config_get_debug_sync(self.internal_builder_config)}
    }

    pub fn set_sparse_weights_mode(&self,mode:bool){
        unsafe {builder_config_set_sparse_weights(self.internal_builder_config,mode)}
    }

    pub fn get_sparse_weights_mode(&self) -> bool{
        unsafe {builder_config_get_sparse_weights(self.internal_builder_config)}
    }

    pub fn set_disable_timing_cache(&self, mode:bool ){
        unsafe{builder_config_set_disable_timing_cache(self.internal_builder_config,mode)}
    }

    pub fn get_disable_timing_cache(&self) -> bool{
        unsafe {builder_config_get_disable_timing_cache(self.internal_builder_config)}
    }

    pub fn set_safety_scope(&self, mode:bool){
        unsafe {builder_config_set_safety_scope(self.internal_builder_config,mode)}
    }

    pub fn get_safety_scope(&self) -> bool{
        unsafe {builder_config_get_safety_scope(self.internal_builder_config)}
    }

    pub fn set_tf32(&self,mode:bool){
        unsafe {builder_config_set_tf32(self.internal_builder_config,mode)}
    }

    pub fn get_tf32(&self) -> bool {
        unsafe {builder_config_get_tf32(self.internal_builder_config)}
    }

    pub fn set_engine_capability(&self, engine_capability: EngineCapability) {
        unsafe { builder_config_set_engine_capability(self.internal_builder_config, engine_capability as i32) }
    }

    pub fn get_engine_capability(&self) -> EngineCapability {
        let primitive = unsafe { builder_config_get_engine_capability(self.internal_builder_config) };
        FromPrimitive::from_i32(primitive).unwrap()

    }



    pub fn reset(&self) {
        unsafe { builder_reset(self.internal_builder)};
        unsafe {builder_config_reset(self.internal_builder_config)};
    }
}

impl<'a> Drop for Builder<'a> {
    fn drop(&mut self) {
        unsafe { destroy_builder(self.internal_builder) };
    }
}
