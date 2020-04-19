use failure::Fail;

#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "invalid hyperparameter: {}", reason)]
    InvalidHyperparameter { reason: String },
    #[fail(display = "invalid data: {}", reason)]
    InvalidData { reason: String },
    #[fail(display = "invalid error: {}", reason)]
    InternalError { reason: String },
}
