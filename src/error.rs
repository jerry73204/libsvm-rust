//! Error types.

use failure::Fail;

/// The error type used across this crate.
#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "invalid hyperparameter: {}", reason)]
    InvalidHyperparameter { reason: String },
    #[fail(display = "invalid data: {}", reason)]
    InvalidData { reason: String },
    #[fail(display = "invalid error: {}", reason)]
    InternalError { reason: String },
}
