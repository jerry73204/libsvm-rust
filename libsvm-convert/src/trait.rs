pub(crate) trait SvmTryFrom<T>: Sized {
    type Error;
    fn svm_try_from(from: T) -> Result<Self, Self::Error>;
}

pub(crate) trait SvmTryInto<T>: Sized {
    type Error;
    fn svm_try_into(self) -> Result<T, Self::Error>;
}
