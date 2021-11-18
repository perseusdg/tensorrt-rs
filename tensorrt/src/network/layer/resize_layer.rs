use std::os::raw::c_float;
use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{resize_layer_set_output_dimensions, resize_layer_get_output_dimensions, resize_layer_set_scales, resize_layer_get_scales, resize_layer_set_resize_mode, resize_layer_get_resize_mode, resize_layer_set_coordinate_transformation, resize_layer_get_coordinate_transformation, resize_layer_set_selector_for_single_pixel, resize_layer_get_selector_for_single_pixel, resize_layer_set_nearest_rounding, resize_layer_get_nearest_rounding, resize_Layer_set_input, nvinfer1_IResizeLayer, nvinfer1_Dims};
use crate::dims::{Dim, Dims};

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ResizeMode {
    NEAREST,
    LINEAR,
}

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ResizeCoordinateTransformation {
    AlignCorners,
    Asymmetric,
    HalfPixel,
}

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ResizeSelector {
    FORMULA,
    UPPER,
}

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum ResizeRoundMode {
    HalfUp,
    HalfDown,
    Floor,
    Ceil,
}

#[derive(Layer)]
pub struct ResizeLayer{
    pub(crate) internal_layer: *mut nvinfer1_IResizeLayer,
}

impl ResizeLayer{
    pub fn set_output_dimensions<T:Dim> (&self,dimensions:T){
        unsafe { resize_layer_set_output_dimensions(self.internal_layer,dimensions.get_internal_dims()) }
    }

    pub fn get_output_dimensions (&self) -> Dims {
        let raw = unsafe { resize_layer_get_output_dimensions(self.internal_layer) };
        Dims(raw)
    }

    pub fn set_scales(&self, nb_scales: i32, scales: Vec<f32>) {
        unsafe {resize_layer_set_scales(self.internal_layer, scales.as_ptr() as *const c_float, nb_scales)}
    }

    pub fn get_scales(&self, size: i32, scales: Vec<f32>) -> i32 {
        let primitive = unsafe { resize_layer_get_scales(self.internal_layer,size,scales.as_ptr() as *mut c_float) };
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_resize_mode(&self,mode:ResizeMode) {
        unsafe { resize_layer_set_resize_mode(self.internal_layer,mode as i32)}
    }

    pub fn get_resize_mode(&self) -> ResizeMode {
        let raw = unsafe { resize_layer_get_resize_mode(self.internal_layer) };
        FromPrimitive::from_i32(raw).unwrap()
    }

    pub fn set_coordinate_transformation(&self, coord_transforms:ResizeCoordinateTransformation) {
        unsafe { resize_layer_set_coordinate_transformation(self.internal_layer, coord_transforms as i32) }
    }

    pub fn get_coordinate_transformation(&self) -> ResizeCoordinateTransformation {
        let raw = unsafe{ resize_layer_get_coordinate_transformation(self.internal_layer) };
        FromPrimitive::from_i32(raw).unwrap()
    }

    pub fn set_selector_for_single_pixel(&self,selector:ResizeSelector) {
        unsafe { resize_layer_set_selector_for_single_pixel(self.internal_layer,selector as i32) }
    }

    pub fn get_selector_for_single_pixel(&self) -> ResizeSelector {
        let raw = unsafe { resize_layer_get_selector_for_single_pixel(self.internal_layer) };
        FromPrimitive::from_i32(raw).unwrap()
    }

    pub fn set_rounding_mode(&self,mode:ResizeRoundMode) {
        unsafe { resize_layer_set_nearest_rounding(self.internal_layer,mode as i32) }
    }

    pub fn get_rounding_mode(&self) -> ResizeRoundMode {
        let raw = unsafe { resize_layer_get_nearest_rounding(self.internal_layer) };
        FromPrimitive::from_i32(raw).unwrap()
    }

    pub fn set_input(&self,index : i32 ,tensor:Tensor) {
        unsafe { resize_Layer_set_input(self.internal_layer,index,tensor.internal_tensor)}
    }


}