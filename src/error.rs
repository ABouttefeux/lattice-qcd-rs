//! defines different error types.

use core::fmt::{Debug, Display};
use std::error::Error;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::thread::ThreadError;

/// Type that can never be (safly) initialized.
/// This is temporary, until [`never`](https://doc.rust-lang.org/std/primitive.never.html) is accepted into stable rust.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum Never {}

impl core::fmt::Display for Never {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for Never {}

/// Errors in the implementation of the library. This is unwanted to return this type but
/// somethimes this is better to return that instead of panicking. It is also used in some example.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum ImplementationError {
    /// We atain a portion of the code that was tought to be unreachable.
    Unreachable,
    /// An option contained an unexpected None value.
    ///
    /// Used when needing to retrun a dyn Error but [`std::option::NoneError`] does not implement [`Error`]
    OptionWithUnexpectedNone,
}

impl Display for ImplementationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImplementationError::Unreachable => {
                write!(f, "internal error: entered unreachable code")
            }
            ImplementationError::OptionWithUnexpectedNone => {
                write!(f, "an option contained an unexpected None value")
            }
        }
    }
}

impl Error for ImplementationError {}

/// Error return while doing multiple steps.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum MultiIntegrationError<Error> {
    /// atempting to integrate doing zero steps
    ZeroIntegration,
    /// An intgration error occured at the position of the first field.
    IntegrationError(usize, Error),
}

impl<Error: Display> Display for MultiIntegrationError<Error> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MultiIntegrationError::ZeroIntegration => write!(f, "error: no integration steps"),
            MultiIntegrationError::IntegrationError(index, error) => {
                write!(f, "error during intgration step {}: {}", index, error)
            }
        }
    }
}

impl<E: Display + Debug + Error + 'static> Error for MultiIntegrationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            MultiIntegrationError::ZeroIntegration => None,
            MultiIntegrationError::IntegrationError(_, error) => Some(error),
        }
    }
}

/// Error while initialising a state
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Copy, Eq)]
pub enum StateInitializationError {
    /// The parameter given for the normal distribution is incorrect.
    InvalideParameterNormalDistribution(rand_distr::NormalError),
    /// Size of lattice and data are imcompatible
    IncompatibleSize,
    /// [`LatticeInitializationError`]
    LatticeInitializationError(LatticeInitializationError),
}

impl From<rand_distr::NormalError> for StateInitializationError {
    fn from(err: rand_distr::NormalError) -> Self {
        Self::InvalideParameterNormalDistribution(err)
    }
}

impl From<LatticeInitializationError> for StateInitializationError {
    fn from(err: LatticeInitializationError) -> Self {
        Self::LatticeInitializationError(err)
    }
}

impl Display for StateInitializationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalideParameterNormalDistribution(error) => {
                write!(f, "normal distribution error: {}", error)
            }
            Self::IncompatibleSize => write!(f, "size of lattice and data are imcompatible"),
            Self::LatticeInitializationError(err) => {
                write!(f, "lattice Initialization error : {}", err)
            }
        }
    }
}

impl Error for StateInitializationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalideParameterNormalDistribution(error) => Some(error),
            Self::IncompatibleSize => None,
            Self::LatticeInitializationError(err) => Some(err),
        }
    }
}

/// Error while initialising a state
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ThreadedStateInitializationError {
    /// multithreading error, see [`ThreadError`].
    ThreadingError(ThreadError),
    /// Other Error cause in non threaded section
    StateInitializationError(StateInitializationError),
}

impl From<ThreadError> for ThreadedStateInitializationError {
    fn from(err: ThreadError) -> Self {
        Self::ThreadingError(err)
    }
}

impl From<StateInitializationError> for ThreadedStateInitializationError {
    fn from(err: StateInitializationError) -> Self {
        Self::StateInitializationError(err)
    }
}

impl Display for ThreadedStateInitializationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ThreadingError(error) => write!(f, "thread error: {}", error),
            Self::StateInitializationError(error) => write!(f, "{}", error),
        }
    }
}

impl Error for ThreadedStateInitializationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ThreadingError(error) => Some(error),
            Self::StateInitializationError(error) => Some(error),
        }
    }
}

/// Error while initialising a lattice
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Copy, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum LatticeInitializationError {
    /// `size` must be stricly greater 0 than and be a finite number.
    NonPositiveSize,
    /// `dim` must be greater than 2.
    DimTooSmall,
    /// the dimension parameter `D = 0` is not valide.
    ZeroDimension,
}

impl Display for LatticeInitializationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveSize => write!(
                f,
                "`size` must be stricly greater than 0 and be a finite number"
            ),
            Self::DimTooSmall => write!(f, "`dim` must be greater or equal to 2"),
            Self::ZeroDimension => write!(f, "the dimension parameter `D = 0` is not valid"),
        }
    }
}

impl Error for LatticeInitializationError {}

/// A struct that combine an error with a owned value
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ErrorWithOnwnedValue<Error, State> {
    error: Error,
    owned: State,
}

impl<Error, State> ErrorWithOnwnedValue<Error, State> {
    getter!(
        /// getter on the error
        pub const error() -> Error
    );

    getter!(
        /// getter on the owned value
        pub const owned() -> State
    );

    /// Create a new Self with an error and an owned value
    pub const fn new(error: Error, owned: State) -> Self {
        Self { error, owned }
    }

    /// Deconstruct the structur.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn deconstruct(self) -> (Error, State) {
        (self.error, self.owned)
    }

    /// Deconstruct the structur returning the error and discarding the owned value.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn error_owned(self) -> Error {
        self.error
    }
}

impl<Error, State> From<(Error, State)> for ErrorWithOnwnedValue<Error, State> {
    fn from(data: (Error, State)) -> Self {
        Self::new(data.0, data.1)
    }
}

impl<Error: Display, State: Display> Display for ErrorWithOnwnedValue<Error, State> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "error {} with data {}", self.error, self.owned)
    }
}

impl<E: Display + Error + Debug + 'static, State: Display + Debug> Error
    for ErrorWithOnwnedValue<E, State>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}

impl<State> From<ErrorWithOnwnedValue<StateInitializationError, State>>
    for StateInitializationError
{
    fn from(data: ErrorWithOnwnedValue<StateInitializationError, State>) -> Self {
        data.error
    }
}

impl<Error, Data1, Data2> From<ErrorWithOnwnedValue<Error, (Data1, Data2)>>
    for ErrorWithOnwnedValue<Error, Data1>
{
    fn from(data: ErrorWithOnwnedValue<Error, (Data1, Data2)>) -> Self {
        Self::new(data.error, data.owned.0)
    }
}
