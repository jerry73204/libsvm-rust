//! The marker types to mark the state of [Svm](crate::model::Svm) type.

use crate::data::SvmNodes;
use std::ptr::NonNull;

/// Used for the trait bound of state types.
pub trait SvmState {}

/// The untrained state marker.
#[derive(Debug)]
pub struct Untrained {
    pub(crate) gamma_opt: Option<f64>,
    pub(crate) params: libsvm_sys::svm_parameter,
    pub(crate) weight_labels: Vec<i32>,
    pub(crate) weights: Vec<f64>,
}

impl SvmState for Untrained {}

/// The trained state marker.
#[derive(Debug)]
pub struct Trained {
    pub(crate) model_ptr: NonNull<libsvm_sys::svm_model>,
    // libsvm_sys::svm_model refers to this struct internally
    pub(crate) nodes_opt: Option<SvmNodes>,
}

impl SvmState for Trained {}

impl Drop for Trained {
    fn drop(&mut self) {
        unsafe {
            if self.model_ptr.as_ref().free_sv != 0 {
                let mut model_ptr_ptr: *mut libsvm_sys::svm_model = self.model_ptr.as_mut();
                libsvm_sys::svm_free_and_destroy_model(
                    &mut model_ptr_ptr as *mut *mut libsvm_sys::svm_model,
                );
            } else {
                libsvm_sys::svm_free_model_content(self.model_ptr.as_ptr());
            }
        }
    }
}
