use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{slice_layer_get_mode, slice_layer_set_mode, slice_layer_set_size, slice_layer_get_size, slice_layer_get_start, slice_layer_set_start, slice_layer_get_stride, slice_layer_set_stride, slice_layer_set_input, nvinfer1_ISliceLayer, nvinfer1_ITensor, nvinfer1_Dims};

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum SliceMode {
    DEFAULT,
    WRAP,
    CLAMP,
    FILL,
    REFLECT,
}

#[derive(Layer)]
pub struct SliceLayer{
    pub(crate) internal_layer: *mut nvinfer1_ISliceLayer
}

impl SliceLayer {
    pub fn set_start(&self, start : nvinfer1_Dims) {
        unsafe { slice_layer_set_start(self.internal_layer,start) }
    }

    pub fn get_start(&self) -> nvinfer1_Dims {
        unsafe { slice_layer_get_start(self.internal_layer) }
    }

    pub fn set_size(&self,size : nvinfer1_Dims) {
        unsafe { slice_layer_set_size(self.internal_layer,size) }
    }

    pub fn get_size(&self) -> nvinfer1_Dims {
        unsafe { slice_layer_get_size(self.internal_layer) }
    }

    pub fn set_stride(&self,stride : nvinfer1_Dims) {
        unsafe { slice_layer_set_stride(self.internal_layer,stride)}
    }

    pub fn get_stride(&self) -> nvinfer1_Dims {
        unsafe { slice_layer_get_stride(self.internal_layer) }
    }

    pub fn set_mode(&self,mode : SliceMode){
        unsafe { slice_layer_set_mode(self.internal_layer,mode as i32) }
    }

    pub fn get_mode(&self) -> SliceMode {
        let primitive = unsafe { slice_layer_get_mode(self.internal_layer) };
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_input(&self,input : Tensor,index : i32){
        unsafe { slice_layer_set_input(self.internal_layer,index,input.internal_tensor)}
    }

}