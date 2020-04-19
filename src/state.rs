use std::ptr::NonNull;

pub trait SvmState {
    // fn gamma(&self) -> Option<f64>;
}

pub struct Untrained {
    pub(crate) gamma_opt: Option<f64>,
    pub(crate) params: libsvm_sys::svm_parameter,
    pub(crate) weight_labels: Vec<i32>,
    pub(crate) weights: Vec<f64>,
}

impl SvmState for Untrained {
    // fn gamma(&self) -> Option<f64> {
    //     self.gamma_opt.clone()
    // }
}

pub struct Trained {
    pub(crate) model_ptr: NonNull<libsvm_sys::svm_model>,
}

impl SvmState for Trained {
    // fn gamma(&self) -> Option<f64> {
    //     Some(self.gamma)
    // }
}

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
