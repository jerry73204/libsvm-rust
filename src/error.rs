//! Error types.

use std::path::PathBuf;

/// The error type used across this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot convert path \"{path}\" to bytes")]
    UnsupportedPath { path: PathBuf },
    #[error("invalid hyperparameter: {reason}")]
    InvalidHyperparameter { reason: String },
    #[error("invalid data: {reason}")]
    InvalidData { reason: String },
    #[error("invalid error: {reason}")]
    InternalError { reason: String },
    #[error("invalid line: {reason}")]
    InvalidLine { reason: String },
}
