
use super::thread::ThreadError;

pub mod state;
pub mod monte_carlo;

pub use state::*;
pub use monte_carlo::*;


/// Error returned by simulation.
#[derive(Debug)]
pub enum SimulationError {
    /// multithreading error, see [`ThreadError`].
    ThreadingError(ThreadError),
    /// Error while initialising.
    InitialisationError,
    /// try to do a simulation with zero steps
    ZeroStep,
    ///
    NotValide,
}

impl From<ThreadError> for SimulationError{
    fn from(err: ThreadError) -> Self{
        SimulationError::ThreadingError(err)
    }
}
