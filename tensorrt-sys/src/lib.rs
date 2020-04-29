/* automatically generated by rust-bindgen */

pub const true_: u32 = 1;
pub const false_: u32 = 0;
pub const __bool_true_false_are_defined: u32 = 1;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Logger {
    _unused: [u8; 0],
}
pub type Logger_t = Logger;
extern "C" {
    pub fn get_tensorrt_version(string: *mut ::std::os::raw::c_char);
}
extern "C" {
    pub fn create_logger() -> *mut Logger_t;
}
extern "C" {
    pub fn log_error(logger: *mut Logger_t, err: *mut ::std::os::raw::c_char);
}
extern "C" {
    pub fn delete_logger(logger: *mut Logger_t);
}
pub type wchar_t = ::std::os::raw::c_int;
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone)]
pub struct max_align_t {
    pub __clang_max_align_nonce1: ::std::os::raw::c_longlong,
    pub __bindgen_padding_0: u64,
    pub __clang_max_align_nonce2: u128,
}
#[test]
fn bindgen_test_layout_max_align_t() {
    assert_eq!(
        ::std::mem::size_of::<max_align_t>(),
        32usize,
        concat!("Size of: ", stringify!(max_align_t))
    );
    assert_eq!(
        ::std::mem::align_of::<max_align_t>(),
        16usize,
        concat!("Alignment of ", stringify!(max_align_t))
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<max_align_t>())).__clang_max_align_nonce1 as *const _ as usize
        },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(max_align_t),
            "::",
            stringify!(__clang_max_align_nonce1)
        )
    );
    assert_eq!(
        unsafe {
            &(*(::std::ptr::null::<max_align_t>())).__clang_max_align_nonce2 as *const _ as usize
        },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(max_align_t),
            "::",
            stringify!(__clang_max_align_nonce2)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Context {
    _unused: [u8; 0],
}
pub type Context_t = Context;
extern "C" {
    pub fn destroy_excecution_context(execution_context: *mut Context_t);
}
extern "C" {
    pub fn context_set_name(execution_context: *mut Context_t, name: *const ::std::os::raw::c_char);
}
extern "C" {
    pub fn context_get_name(execution_context: *mut Context_t) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn execute(
        execution_context: *const Context_t,
        input_data: *const f32,
        input_data_size: usize,
        input_index: ::std::os::raw::c_int,
        output_data: *mut f32,
        output_size: usize,
        output_index: ::std::os::raw::c_int,
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HostMemory {
    _unused: [u8; 0],
}
pub type HostMemory_t = HostMemory;
extern "C" {
    pub fn destroy_host_memory(host_memory: *mut HostMemory_t);
}
extern "C" {
    pub fn host_memory_get_data(host_memory: *mut HostMemory_t) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn host_memory_get_size(host_memory: *mut HostMemory_t) -> usize;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Engine {
    _unused: [u8; 0],
}
pub type Engine_t = Engine;
extern "C" {
    pub fn destroy_cuda_engine(engine: *mut Engine_t);
}
extern "C" {
    pub fn engine_create_execution_context(engine: *mut Engine_t) -> *mut Context_t;
}
extern "C" {
    pub fn get_nb_bindings(engine: *mut Engine_t) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn get_binding_name(
        engine: *mut Engine_t,
        binding_index: ::std::os::raw::c_int,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    pub fn get_binding_index(
        engine: *mut Engine_t,
        op_name: *const ::std::os::raw::c_char,
    ) -> ::std::os::raw::c_int;
}
extern "C" {
    pub fn engine_serialize(engine: *mut Engine_t) -> *mut HostMemory_t;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Runtime {
    _unused: [u8; 0],
}
pub type Runtime_t = Runtime;
extern "C" {
    pub fn create_infer_runtime(logger: *mut Logger_t) -> *mut Runtime_t;
}
extern "C" {
    pub fn deserialize_cuda_engine(
        runtime: *mut Runtime_t,
        blob: *const ::std::os::raw::c_void,
        size: ::std::os::raw::c_ulonglong,
    ) -> *mut Engine_t;
}
extern "C" {
    pub fn destroy_infer_runtime(runtime: *mut Runtime_t);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Dims {
    pub nbDims: ::std::os::raw::c_int,
    pub d: *mut ::std::os::raw::c_int,
    pub type_: *mut ::std::os::raw::c_int,
}
#[test]
fn bindgen_test_layout_Dims() {
    assert_eq!(
        ::std::mem::size_of::<Dims>(),
        24usize,
        concat!("Size of: ", stringify!(Dims))
    );
    assert_eq!(
        ::std::mem::align_of::<Dims>(),
        8usize,
        concat!("Alignment of ", stringify!(Dims))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<Dims>())).nbDims as *const _ as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(Dims),
            "::",
            stringify!(nbDims)
        )
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<Dims>())).d as *const _ as usize },
        8usize,
        concat!("Offset of field: ", stringify!(Dims), "::", stringify!(d))
    );
    assert_eq!(
        unsafe { &(*(::std::ptr::null::<Dims>())).type_ as *const _ as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(Dims),
            "::",
            stringify!(type_)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Network {
    _unused: [u8; 0],
}
pub type Network_t = Network;
extern "C" {
    pub fn destroy_network(network: *mut Network_t);
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct UffParser {
    _unused: [u8; 0],
}
pub type UffParser_t = UffParser;
extern "C" {
    pub fn uffparser_create_uff_parser() -> *mut UffParser_t;
}
extern "C" {
    pub fn uffparser_destroy_uff_parser(uff_parser: *mut UffParser_t);
}
extern "C" {
    pub fn uffparser_register_input(
        uff_parser: *const UffParser_t,
        input_name: *const ::std::os::raw::c_char,
        input_dims: Dims,
    ) -> bool;
}
extern "C" {
    pub fn uffparser_register_output(
        uff_parser: *const UffParser_t,
        output_name: *const ::std::os::raw::c_char,
    ) -> bool;
}
extern "C" {
    pub fn uffparser_parse(
        uff_parser: *const UffParser_t,
        file: *const ::std::os::raw::c_char,
        network: *const Network_t,
    ) -> bool;
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Builder {
    _unused: [u8; 0],
}
pub type Builder_t = Builder;
extern "C" {
    pub fn create_infer_builder(logger: *mut Logger_t) -> *mut Builder_t;
}
extern "C" {
    pub fn destroy_builder(builder: *mut Builder_t);
}
extern "C" {
    pub fn create_network(builder: *mut Builder_t) -> *mut Network_t;
}
extern "C" {
    pub fn build_cuda_engine(builder: *mut Builder_t, network: *mut Network_t) -> *mut Engine_t;
}
