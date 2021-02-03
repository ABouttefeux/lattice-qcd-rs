
//! Numerical integrators to carry out simulations.

use super::{
    simulation::{
        SimulationError,
        LatticeHamiltonianSimulationState,
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
use na::Vector3;

pub mod symplectic_euler;
pub mod symplectic_euler_rayon;

pub use symplectic_euler::*;
pub use symplectic_euler_rayon::*;


/// Define an numerical integrator
pub trait Integrator<State, State2>
    where State: LatticeHamiltonianSimulationState,
    State2: LatticeHamiltonianSimulationState,
{
    /// Do one simulation step
    fn integrate(&self, l: &State, delta_t: Real) ->  Result<State2, SimulationError>;
}

/// Define an symplectic numerical integrator
pub trait SymplecticIntegrator<State, State2>
    where Self:Integrator<State, State2>,
    State: LatticeHamiltonianSimulationState,
    State2: LatticeHamiltonianSimulationState,
{}
    
/// function for link intregration
fn integrate_link<State>(link: &LatticeLinkCanonical, l: &State, delta_t: Real) -> CMatrix3
    where State: LatticeHamiltonianSimulationState,
{
    let canonical_link = LatticeLink::from(*link);
    let initial_value = l.link_matrix().get_matrix(&canonical_link, l.lattice()).unwrap();
    initial_value + l.get_derivatives_u(link).unwrap() * Complex::from(delta_t)
}

/// function for "Electrical" field intregration
fn integrate_efield<State>(point: &LatticePoint, l: &State, delta_t: Real) -> Vector3<Su3Adjoint>
    where State: LatticeHamiltonianSimulationState,
{
    let initial_value = *l.e_field().get_e_vec(point, l.lattice()).unwrap();
    let deriv = l.get_derivative_e(point).unwrap();
    initial_value + deriv.map(|el| el * delta_t)
}
