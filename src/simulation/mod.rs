
//! Simulation module. Containe Monte Carlo algorithms and simulation states.
//!
// TODO more doc

use super::thread::ThreadError;

pub mod state;
pub mod monte_carlo;

pub use state::*;
pub use monte_carlo::*;


/// Error returned by simulation.
#[derive(Debug)]
#[non_exhaustive]
pub enum SimulationError {
    /// multithreading error, see [`ThreadError`].
    ThreadingError(ThreadError),
    /// Error while initialising.
    InitialisationError,
    /// try to do a simulation with zero steps.
    ZeroStep,
    /// What ever you did it wasn't valide.
    NotValide,
    /// The parameter given are incorrect.
    InvalideParameterDistribution(rand_distr::NormalError), // TODO improve
}

/// Returns [`SimulationError::InvalideParameterDistribution`] (err).
impl From<rand_distr::NormalError> for SimulationError {
    fn from(err: rand_distr::NormalError) -> Self{
        SimulationError::InvalideParameterDistribution(err)
    }
}

/// Conversion from [`ThreadError`] to [`SimulationError::ThreadingError`].
/// used for simplification in the usage of `?` operator.
impl From<ThreadError> for SimulationError {
    fn from(err: ThreadError) -> Self{
        SimulationError::ThreadingError(err)
    }
}
