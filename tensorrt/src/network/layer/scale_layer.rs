use super::*;
use tensorrt_rs_derive::Layer;
use std::os::raw::c_void;
use num_traits::pow;
use crate::weights::*;
use tensorrt_sys::{
    scale_layer_get_mode,scale_layer_set_mode,scale_layer_set_power,
    scale_layer_get_power,scale_layer_set_shift,scale_layer_get_shift,
    scale_layer_set_channel_axis,scale_layer_get_channel_axis,nvinfer1_IScaleLayer,
    nvinfer1_Weights,
};

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ScaleMode {
    UNIFORM,
    CHANNEL,
    ELEMENTWISE,
}

#[derive(Layer)]
pub struct ScaleLayer{
    pub(crate) internal_layer: *mut nvinfer1_IScaleLayer,
}

impl ScaleLayer {
    pub fn set_mode(&self,mode : ScaleMode) {
        unsafe {scale_layer_set_mode(self.internal_layer,mode as i32)}
    }

    pub fn get_mode(&self) -> ScaleMode {
        let primitive = unsafe {scale_layer_get_mode(self.internal_layer)};
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_shift(&self,weights:nvinfer1_Weights) {
        unsafe { scale_layer_set_shift(self.internal_layer,weights)}
    }

    pub fn get_shift(&self) -> nvinfer1_Weights {
        unsafe{scale_layer_get_shift(self.internal_layer)}
    }

    pub fn set_power(&self,power:nvinfer1_Weights) {
        unsafe { scale_layer_set_power(self.internal_layer,power)}
    }

    pub fn get_power(&self) -> nvinfer1_Weights {
        unsafe {scale_layer_get_power(self.internal_layer)}
    }

    pub fn set_channel_axis(&self,channel_axis : i32){
        unsafe { scale_layer_set_channel_axis(self.internal_layer,channel_axis) }
    }

    pub fn get_channel_axis(&self) -> i32 {
        unsafe { scale_layer_get_channel_axis(self.internal_layer) as i32 }
    }

}