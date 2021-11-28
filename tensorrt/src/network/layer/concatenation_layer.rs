use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{concatenation_layer_get_axis, concatenation_layer_set_axis, nvinfer1_IConcatenationLayer};

#[derive(Layer)]
pub struct ConcatenationLayer{
    pub(crate) internal_layer: *mut nvinfer1_IConcatenationLayer,
}

impl ConcatenationLayer{
    pub fn set_axis(&self,axis : i32){
        unsafe {concatenation_layer_set_axis(self.internal_layer,axis)}
    }

    pub fn get_axis(&self) -> i32{
        unsafe {concatenation_layer_get_axis(self.internal_layer)}
    }
}
