use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{scatter_layer_get_axis, scatter_layer_set_axis, scatter_layer_get_mode, scatter_layer_set_mode, nvinfer1_IScatterLayer};

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ScatterMode {
    ELEMENT,
    ND,
}

#[derive(Layer)]
pub struct ScatterLayer{
    pub(crate) internal_layer: *mut nvinfer1_IScatterLayer,
}

impl ScatterLayer{
    pub fn set_mode(&self,mode:ScatterMode) {
        unsafe {scatter_layer_set_mode(self.internal_layer,mode as i32)}
    }

    pub fn get_mode(&self) -> ScatterMode {
        let primitive = unsafe{scatter_layer_get_mode(self.internal_layer)};
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_axis(&self,axis:i32) {
        unsafe {scatter_layer_set_axis(self.internal_layer,axis)}
    }

    pub fn get_axis(&self) -> i32 {
        unsafe {scatter_layer_get_axis(self.internal_layer)}
    }

}