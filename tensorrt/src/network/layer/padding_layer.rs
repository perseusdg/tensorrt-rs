use super::*;
use tensorrt_rs_derive::Layer;
use tensorrt_sys::{padding_layer_set_pre_padding, padding_layer_get_pre_padding, padding_layer_set_post_padding, padding_layer_get_post_padding, nvinfer1_IPaddingLayer, nvinfer1_Dims};
use crate::dims::Dims;

#[derive(Layer)]
pub struct PaddingLayer{
    pub(crate) internal_layer : *mut nvinfer1_IPaddingLayer,
}

impl PaddingLayer {
    pub fn set_pre_padding(&self, dimensions:nvinfer1_Dims) {
        unsafe {padding_layer_set_pre_padding(self.internal_layer,dimensions)}
    }

    pub fn get_pre_padding(&self) -> nvinfer1_Dims {
        unsafe {padding_layer_get_pre_padding(self.internal_layer)}
    }

    pub fn set_post_padding(&self, dimensions:nvinfer1_Dims) {
        unsafe {padding_layer_set_post_padding(self.internal_layer,dimensions)}
    }

    pub fn get_post_padding(&self) -> nvinfer1_Dims {
        unsafe {padding_layer_get_post_padding(self.internal_layer)}
    }

}
