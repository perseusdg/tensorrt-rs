use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{reduce_layer_set_operation, reduce_layer_get_operation, reduce_layer_set_reduce_axis, reduce_layer_get_reduce_axis, reduce_layer_set_keep_dimensions, reduce_layer_get_keep_dimensions, nvinfer1_IReduceLayer};

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ReduceOperation {
    SUM,
    PROD,
    MAX,
    MIN,
    AVG,
}


#[derive(Layer)]
pub struct ReduceLayer{
    pub(crate) internal_layer: *mut nvinfer1_IReduceLayer,
}

impl ReduceLayer{
    pub fn set_operation(&self,op:ReduceOperation) {
        unsafe { reduce_layer_set_operation(self.internal_layer,op as i32) }
    }

    pub fn get_operation(&self) -> ReduceOperation {
        let primitive = unsafe {reduce_layer_get_operation(self.internal_layer)};
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_reduce_axis(&self,reduce_axes:u32) {
        unsafe { reduce_layer_set_reduce_axis(self.internal_layer,reduce_axes) }
    }

    pub fn get_reduce_axis(&self) -> u32 {
        unsafe { reduce_layer_get_reduce_axis(self.internal_layer) }
    }

    pub fn set_keep_dimensions(&self,keep:bool) {
        unsafe { reduce_layer_set_keep_dimensions(self.internal_layer,keep) }
    }

    pub fn get_keep_dimensions(&self) -> bool {
        unsafe { reduce_layer_get_keep_dimensions(self.internal_layer) }
    }
}