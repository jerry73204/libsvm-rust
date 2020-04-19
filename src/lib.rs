//! High-level and rusty bindings for libsvm.

#![feature(const_generics)]
#![feature(fixed_size_array)]

pub mod consts;
pub mod data;
pub mod error;
pub mod init;
pub mod model;
pub mod state;

pub use data::SvmNodes;
pub use error::Error;
pub use init::{KernelInit, ModelInit, SvmInit};
pub use model::Svm;
pub use state::{SvmState, Trained, Untrained};
