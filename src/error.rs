
//! defines different error types.

use std::error::Error;

/// Type that can never be (safly) initialized.
/// This is temporary, until [`never`](https://doc.rust-lang.org/std/primitive.never.html) is accepted into stable rust.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum Never {}

impl core::fmt::Display for Never {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for Never {}

/// Errors in the implementation of the library. This is unwanted to return this type but
/// somethimes this is better to return that instead of panicking.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ImplementationError {
    /// We atain a portion of the code that was tought to be unreachable.
    Unreachable,
}

/// Error return while doing multiple steps.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MultiIntegrationError<Error, State> {
    /// atempting to integrate doing zero steps
    ZeroIntegration,
    /// An intgration error occured at the position of the first field, return an owned value ofthe
    /// integration until the error occured so that everything isn't loss
    IntegrationError(usize, Error, Option<State>),
    /// an [`ImplementationError`] occured.
    ImplementationError(ImplementationError),
}
