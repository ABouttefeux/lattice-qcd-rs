//! Defines different error types.

// TODO unwrap, panic and expect => error make a issue on git

use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::thread::ThreadError;

/// A type that can never be (safely) initialized.
/// This is temporary, until [`never`](https://doc.rust-lang.org/std/primitive.never.html)
/// is accepted into stable rust.
///
/// # Example
/// ```
/// # use lattice_qcd_rs::error::Never;
/// trait Something {
///     type Error;
///
///     fn do_something(&self) -> Result<(), Self::Error>;
/// }
///
/// struct SomethingImpl;
///
/// impl Something for SomethingImpl {
///     type Error = Never; // We never have an error
///
///     fn do_something(&self) -> Result<(), Self::Error> {
///         // implementation that never fails
///         Ok(())
///     }
/// }
/// ```
/// the size of [`Never`] is 0
/// ```
/// # use lattice_qcd_rs::error::Never;
/// assert_eq!(std::mem::size_of::<Never>(), 0);
/// // the size is still zero. because () is size 0.
/// assert_eq!(std::mem::size_of::<Result<(), Never>>(), 0);
/// assert_eq!(std::mem::size_of::<Result<u8, Never>>(), 1);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum Never {}

impl Display for Never {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for Never {}

/// Errors in the implementation of the library. This is unwanted to return this type but
/// sometimes this is better to return that instead of panicking. It is also used in some example.
// TODO example
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum ImplementationError {
    /// We attain a portion of the code that was thought to be unreachable.
    Unreachable,
    /// An option contained an unexpected None value.
    ///
    /// Used when needing to return a dyn Error but `std::option::NoneError` does not implement [`Error`]
    // TODO NoneError
    OptionWithUnexpectedNone,
}

impl Display for ImplementationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
// TODO example
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum MultiIntegrationError<Error> {
    /// attempting to integrate doing zero steps
    ZeroIntegration,
    /// An integration error occurred at the position of the first field.
    IntegrationError(usize, Error),
}

impl<Error: Display> Display for MultiIntegrationError<Error> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            MultiIntegrationError::ZeroIntegration => write!(f, "no integration steps"),
            MultiIntegrationError::IntegrationError(index, error) => {
                write!(f, "error during integration step {}: {}", index, error)
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

/// Error while initializing a state
// TODO example
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Copy, Eq)]
pub enum StateInitializationError {
    /// The parameter given for the normal distribution is incorrect.
    InvalidParameterNormalDistribution(rand_distr::NormalError),
    /// Size of lattice and data are incompatible
    IncompatibleSize,
    /// [`LatticeInitializationError`]
    LatticeInitializationError(LatticeInitializationError),
    /// Gauss projection failed
    GaussProjectionError,
}

impl From<rand_distr::NormalError> for StateInitializationError {
    fn from(err: rand_distr::NormalError) -> Self {
        Self::InvalidParameterNormalDistribution(err)
    }
}

impl From<LatticeInitializationError> for StateInitializationError {
    fn from(err: LatticeInitializationError) -> Self {
        Self::LatticeInitializationError(err)
    }
}

impl Display for StateInitializationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameterNormalDistribution(error) => {
                write!(f, "normal distribution error: {}", error)
            }
            Self::IncompatibleSize => write!(f, "size of lattice and data are incompatible"),
            Self::LatticeInitializationError(err) => {
                write!(f, "lattice Initialization error: {}", err)
            }
            Self::GaussProjectionError => write!(f, "gauss projection could not finish"),
        }
    }
}

impl Error for StateInitializationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidParameterNormalDistribution(error) => Some(error),
            Self::IncompatibleSize => None,
            Self::LatticeInitializationError(err) => Some(err),
            Self::GaussProjectionError => None,
        }
    }
}

/// Error while initializing a state using multiple thread or threaded function.
// TODO example
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
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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

/// Error while initializing a lattice
// TODO example
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Copy, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum LatticeInitializationError {
    /// `size` must be strictly greater 0 than and be a finite number.
    NonPositiveSize,
    /// `dim` must be greater than 2.
    DimTooSmall,
    /// the dimension parameter `D = 0` is not valid.
    ZeroDimension,
}

impl Display for LatticeInitializationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveSize => write!(
                f,
                "`size` must be strictly greater than 0 and be a finite number"
            ),
            Self::DimTooSmall => write!(f, "`dim` must be greater or equal to 2"),
            Self::ZeroDimension => write!(f, "the dimension parameter `D = 0` is not valid"),
        }
    }
}

impl Error for LatticeInitializationError {}

/// A struct that combine an error with a owned value
// TODO example / remove
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ErrorWithOwnedValue<Error, State> {
    error: Error,
    owned: State,
}

impl<Error, State> ErrorWithOwnedValue<Error, State> {
    getter!(
        /// Getter on the error.
        pub const error() -> Error
    );

    getter!(
        /// Getter on the owned value.
        pub const owned() -> State
    );

    /// Create a new Self with an error and an owned value
    pub const fn new(error: Error, owned: State) -> Self {
        Self { error, owned }
    }

    /// Deconstruct the structure.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn deconstruct(self) -> (Error, State) {
        (self.error, self.owned)
    }

    /// Deconstruct the structure returning the error and discarding the owned value.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn error_owned(self) -> Error {
        self.error
    }
}

impl<Error, State> From<(Error, State)> for ErrorWithOwnedValue<Error, State> {
    fn from(data: (Error, State)) -> Self {
        Self::new(data.0, data.1)
    }
}

impl<Error: Display, State: Display> Display for ErrorWithOwnedValue<Error, State> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "error {} with data {}", self.error, self.owned)
    }
}

impl<E: Display + Error + Debug + 'static, State: Display + Debug> Error
    for ErrorWithOwnedValue<E, State>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}

impl<State> From<ErrorWithOwnedValue<StateInitializationError, State>>
    for StateInitializationError
{
    fn from(data: ErrorWithOwnedValue<StateInitializationError, State>) -> Self {
        data.error
    }
}

impl<Error, Data1, Data2> From<ErrorWithOwnedValue<Error, (Data1, Data2)>>
    for ErrorWithOwnedValue<Error, Data1>
{
    fn from(data: ErrorWithOwnedValue<Error, (Data1, Data2)>) -> Self {
        Self::new(data.error, data.owned.0)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn implementation_error() {
        assert_eq!(
            ImplementationError::Unreachable.to_string(),
            "internal error: entered unreachable code"
        );
        assert_eq!(
            ImplementationError::OptionWithUnexpectedNone.to_string(),
            "an option contained an unexpected None value"
        );
    }

    /// Test the multi integration error.
    #[test]
    fn multi_integration_error() {
        let e1 = MultiIntegrationError::<LatticeInitializationError>::ZeroIntegration;
        assert_eq!(e1.to_string(), "no integration steps");

        let index = 2;
        let error = LatticeInitializationError::DimTooSmall;
        let e2 = MultiIntegrationError::IntegrationError(index, error);
        assert_eq!(
            e2.to_string(),
            format!("error during integration step {}: {}", index, error)
        );
        assert!(e1.source().is_none());
        assert!(e2.source().is_some());
    }

    #[allow(clippy::missing_const_for_fn)] // cannot test const function
    #[test]
    fn never_size() {
        assert_eq!(std::mem::size_of::<Never>(), 0);
    }
}
