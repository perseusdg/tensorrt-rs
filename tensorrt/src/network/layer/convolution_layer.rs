use super::*;
use tensorrt_rs_derive::Layer;
use crate::dims::*;
use crate::weights::*;
use pooling_layer::PaddingMode;
use tensorrt_sys::{convolution_layer_set_nb_output_maps, convolution_layer_get_nb_output_maps, convolution_layer_set_nb_groups, convolution_layer_get_nb_groups, convolution_layer_set_kernel_weights, convolution_layer_get_kernel_weights, convolution_layer_set_bias_weights, convolution_layer_get_bias_weights, convolution_layer_set_pre_padding, convolution_layer_get_pre_padding, convolution_layer_set_post_padding, convolution_layer_get_post_padding, convolution_layer_set_padding_mode, convolution_layer_get_padding_mode, convolution_layer_set_kernel_size_nd, convolution_layer_get_kernel_size_nd, convolution_layer_set_stride_nd, convolution_layer_get_stride_nd, convolution_layer_set_padding_nd, convolution_layer_get_padding_nd, convolution_layer_set_dilation_nd, convolution_layer_get_dilation_nd, convolution_layer_set_input, nvinfer1_IConvolutionLayer};


#[derive(Layer)]
pub struct ConvolutionLayer{
    pub(crate) internal_layer: *mut nvinfer1_IConvolutionLayer,
}

impl ConvolutionLayer{
    pub fn set_nb_output_maps(&self,nb_output_maps: i32){
        unsafe { convolution_layer_set_nb_output_maps(self.internal_layer,nb_output_maps) }
    }

    pub fn get_nb_output_maps(&self) -> i32 {
        unsafe { convolution_layer_get_nb_output_maps(self.internal_layer) }
    }

    pub fn set_nb_groups(&self,nb_groups: i32){
        unsafe { convolution_layer_set_nb_groups(self.internal_layer,nb_groups) }
    }

    pub fn get_nb_groups(&self) -> i32 {
        unsafe { convolution_layer_get_nb_groups(self.internal_layer) }
    }

    pub fn set_kernel_weights<T:Weight> (&self,kernel_weights : T) {
        unsafe { convolution_layer_set_kernel_weights(self.internal_layer,kernel_weights.get_internal_weights()) }
    }

    pub fn get_kernel_weights(&self) -> Weights {
        let primitive = unsafe { convolution_layer_get_kernel_weights(self.internal_layer) };
        Weights(primitive)
    }

    pub fn set_bias_weights<T:Weight> (&self,bias_weights : T) {
        unsafe { convolution_layer_set_bias_weights(self.internal_layer,bias_weights.get_internal_weights()) }
    }

    pub fn get_bias_weights(&self) -> Weights {
        let primitive = unsafe { convolution_layer_get_bias_weights(self.internal_layer) };
        Weights(primitive)
    }

    pub fn set_pre_padding<T:Dim>(&self,padding : T) {
        unsafe { convolution_layer_set_pre_padding(self.internal_layer,padding.get_internal_dims())}
    }

    pub fn get_pre_padding(&self) -> Dims {
        let primitive = unsafe { convolution_layer_get_pre_padding(self.internal_layer) };
        Dims(primitive)
    }

    pub fn set_post_padding<T:Dim>(&self,padding : T) {
        unsafe { convolution_layer_set_post_padding(self.internal_layer,padding.get_internal_dims()) }
    }

    pub fn get_post_padding(&self) -> Dims {
        let primitive = unsafe { convolution_layer_get_post_padding(self.internal_layer) };
        Dims(primitive)
    }

    pub fn set_padding_mode(&self,padding_mode : PaddingMode) {
        unsafe { convolution_layer_set_padding_mode(self.internal_layer,padding_mode as i32) }
    }

    pub fn get_padding_mode(&self) -> PaddingMode {
        let primitive = unsafe { convolution_layer_get_padding_mode(self.internal_layer) };
        FromPrimitive::from_i32(primitive).unwrap()
    }

    pub fn set_kernel_size_nd<T:Dim>(&self,kernel_size : T) {
        unsafe { convolution_layer_set_kernel_size_nd(self.internal_layer,kernel_size.get_internal_dims()) }
    }

    pub fn get_kernel_size_nd(&self) -> Dims {
        let primitive = unsafe { convolution_layer_get_kernel_size_nd(self.internal_layer) };
        Dims(primitive)
    }

    pub fn set_stride_nd<T:Dim>(&self,stride : T) {
        unsafe { convolution_layer_set_stride_nd(self.internal_layer,stride.get_internal_dims()) }
    }

    pub fn get_stride_nd(&self) -> Dims {
        let primitive = unsafe { convolution_layer_get_stride_nd(self.internal_layer) };
        Dims(primitive)
    }

    pub fn set_padding_nd<T:Dim> (&self,padding : T) {
        unsafe { convolution_layer_set_padding_nd(self.internal_layer,padding.get_internal_dims()) }
    }

    pub fn get_padding_nd(&self) -> Dims {
        let primitive = unsafe { convolution_layer_get_padding_nd(self.internal_layer) };
        Dims(primitive)
    }

    pub fn set_dilation_nd<T:Dim> (&self,dilation : T) {
        unsafe { convolution_layer_set_dilation_nd(self.internal_layer,dilation.get_internal_dims()) }
    }

    pub fn get_dilation_nd(&self) -> Dims {
        let primitive = unsafe { convolution_layer_get_dilation_nd(self.internal_layer)};
        Dims(primitive)
    }

    pub fn set_input(&self,index : i32,input: Tensor) {
        unsafe { convolution_layer_set_input(self.internal_layer,index,input.internal_tensor)}
    }
}