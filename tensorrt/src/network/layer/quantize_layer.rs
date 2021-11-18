use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{quantize_layer_set_axis,quantize_layer_get_axis,nvinfer1_IQuantizeLayer};

#[derive(Layer)]
pub struct QuantizeLayer{
    pub(crate) internal_layer: *mut nvinfer1_IQuantizeLayer,
}

impl QuantizeLayer{
    pub fn set_axis(&self,axis:i32){
        unsafe{quantize_layer_set_axis(self.internal_layer,axis)}
    }

    pub fn get_axis(&self) -> i32 {
        unsafe{quantize_layer_get_axis(self.internal_layer)}
    }
}