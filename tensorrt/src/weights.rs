use std::ffi::c_void;
use tensorrt_sys::{create_weights,nvinfer1_Weights};

mod private{
    pub trait WeightsPrivate{
        fn get_internal_weights(&self) -> super::nvinfer1_Weights;
    }
}


pub trait Weight: private::WeightsPrivate{
    fn count(&self) -> i64 {self.get_internal_weights().count}
    fn data_type(&self) -> i32 {self.get_internal_weights().type_ as i32}
    fn values(&self) -> *const c_void {self.get_internal_weights().values as *mut c_void}
}

#[repr(C)]
pub enum DataType {
    Float,
    Half,
    Int8,
    Int32,
    Bool,
}

#[repr(transparent)]
pub struct Weights(pub nvinfer1_Weights);

impl Weights{
    pub fn new(count : i64,data_type : i32,value : *mut c_void) -> Weights{
        let nv_weights = unsafe{create_weights(data_type as i32,value,count as i64)};
        Weights(nv_weights)
    }
}

impl private::WeightsPrivate for Weights{
    fn get_internal_weights(&self) -> nvinfer1_Weights {
        self.0
    }
}

impl Weight for Weights{}

