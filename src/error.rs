//! Error types.

/// The error type used across this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid hyperparameter: {reason}")]
    InvalidHyperparameter { reason: String },
    #[error("invalid data: {reason}")]
    InvalidData { reason: String },
    #[error("invalid error: {reason}")]
    InternalError { reason: String },
}
