
//! Numerical integrators to carry out simulations.
//!
//! See [`SymplecticIntegrator`]. The simulations are done on [`LatticeHamiltonianSimulationState`]
//! It also require a notion of [`SimulationStateSynchrone`] and [`SimulationStateLeapFrog`].
//!
//! Even thought it is effortless to implement both [`SimulationStateSynchrone`] and [`SimulationStateLeapFrog`].
//! I adivce against it and implement only [`SimulationStateSynchrone`] and use [`super::simulation::SimulationStateLeap`]
//! for leap frog states as it gives a compile time check that you did not forget doing a demi steps.
//!
//! This library gives two implementations of [`SymplecticIntegrator`]: [`SymplecticEuler`] and [`SymplecticEulerRayon`].
//! I would advice using [`SymplecticEulerRayon`] if you do not mind the little more momory it uses.


use super::{
    simulation::{
        SimulationError,
        LatticeHamiltonianSimulationState,
        SimulationStateLeapFrog,
        SimulationStateSynchrone,
    },
    Real,
    Complex,
    lattice::{
        LatticeLinkCanonical,
        LatticeLink,
        LatticePoint
    },
    CMatrix3,
    field::Su3Adjoint,
};
use na::Vector4;

pub mod symplectic_euler;
pub mod symplectic_euler_rayon;

pub use symplectic_euler::SymplecticEuler;
pub use symplectic_euler_rayon::SymplecticEulerRayon;

/*
/// Define an numerical integrator
pub trait Integrator<State, State2>
    where State: LatticeHamiltonianSimulationState,
    State2: LatticeHamiltonianSimulationState,
{
    /// Do one simulation step
    fn integrate(&self, l: &State, delta_t: Real) ->  Result<State2, SimulationError>;
}
*/

/// Define an symplectic numerical integrator
///
/// The integrator evlove the state in time.
///
/// The integrator should be capable of switching between Sync state
/// (q (ok link matrices) at time T , p(or e_field) at time T )
/// and leap frog (a at time T, p at time T + 1/2)
pub trait SymplecticIntegrator<StateSync, StateLeap>
    where StateSync: SimulationStateSynchrone,
    StateLeap: SimulationStateLeapFrog,
{
    
    fn integrate_sync_sync(&self, l: &StateSync, delta_t: Real) ->  Result<StateSync, SimulationError>;
    fn integrate_leap_leap(&self, l: &StateLeap, delta_t: Real) ->  Result<StateLeap, SimulationError>;
    fn integrate_sync_leap(&self, l: &StateSync, delta_t: Real) ->  Result<StateLeap, SimulationError>;
    fn integrate_leap_sync(&self, l: &StateLeap, delta_t: Real) ->  Result<StateSync, SimulationError>;
}

/// function for link intregration
fn integrate_link<State>(link: &LatticeLinkCanonical, l: &State, delta_t: Real) -> CMatrix3
    where State: LatticeHamiltonianSimulationState,
{
    let canonical_link = LatticeLink::from(*link);
    let initial_value = l.link_matrix().get_matrix(&canonical_link, l.lattice()).unwrap();
    initial_value + l.get_derivatives_u(link).unwrap() * Complex::from(delta_t)
}

/// function for "Electrical" field intregration
fn integrate_efield<State>(point: &LatticePoint, l: &State, delta_t: Real) -> Vector4<Su3Adjoint>
    where State: LatticeHamiltonianSimulationState,
{
    let initial_value = *l.e_field().get_e_vec(point, l.lattice()).unwrap();
    let deriv = l.get_derivative_e(point).unwrap();
    initial_value + deriv.map(|el| el * delta_t)
}
