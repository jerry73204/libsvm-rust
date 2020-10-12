//! High-level bindings for libsvm.

#![cfg_attr(feature = "nightly", feature(min_const_generics))]
#![cfg_attr(feature = "nightly", feature(array_methods))]

use std::os::raw::c_char;

#[ctor::ctor]
fn disable_print_in_libsvm() {
    unsafe {
        libsvm_sys::svm_set_print_string_function(Some(noop));
    }
}
unsafe extern "C" fn noop(_: *const c_char) {}

pub mod consts;
pub mod data;
pub mod error;
pub mod init;
pub mod model;

pub use data::SvmNodes;
pub use error::Error;
pub use init::{KernelInit, ModelInit, SvmInit};
pub use model::{SvmPredictor, SvmTrainer};
