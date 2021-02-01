
// TODO this file need cleanup

//! Numerical integrators to carry out simulations.

use super::{
    simulation::{
        SimulationError,
        SimulationState,
    },
    Real,
};

pub mod symplectic_euler;
pub mod symplectic_euler_rayon;

pub use symplectic_euler::*;
pub use symplectic_euler_rayon::*;


/// Define an numerical integrator
pub trait Integrator<State, State2>
    where State: SimulationState,
    State2: SimulationState,
{
    /// Do one simulation step
    fn integrate(&self, l: &State, delta_t: Real) ->  Result<State2, SimulationError>;
}

/// Define an symplectic numerical integrator
pub trait SymplecticIntegrator<State, State2>
    where Self:Integrator<State, State2>,
    State: SimulationState,
    State2: SimulationState,
{}
