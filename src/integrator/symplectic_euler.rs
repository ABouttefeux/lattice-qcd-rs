
//! Basic symplectic Euler integrator

use na::{
    Vector3,
};
use super::{
    super::{
        field::{
            Su3Adjoint,
            EField,
            LinkMatrix,
        },
        thread::{
            run_pool_parallel_vec,
            ThreadError,
        },
        CMatrix3,
        Real,
        simulation::{
            SimulationError,
            LatticeHamiltonianSimulationState,
            SimulationStateSynchrone,
            SimulationStateLeapFrog,
            LatticeHamiltonianSimulationStateNew,
        },
    },
    Integrator,
    SymplecticIntegrator,
    integrate_link,
    integrate_efield,
};
use std::vec::Vec;

fn get_link_matrix_integrate<State> (l: &State, number_of_thread: usize, delta_t: Real) -> Result<Vec<CMatrix3>, ThreadError>
    where State: LatticeHamiltonianSimulationState
{
    run_pool_parallel_vec(
        l.lattice().get_links_space(),
        l,
        &|link, l| integrate_link(link, l, delta_t),
        number_of_thread,
        l.lattice().get_number_of_canonical_links_space(),
        l.lattice(),
        CMatrix3::zeros(),
    )
}

fn get_e_field_integrate<State> (l: &State, number_of_thread: usize, delta_t: Real) -> Result<Vec<Vector3<Su3Adjoint>>, ThreadError>
    where State: LatticeHamiltonianSimulationState
{
    run_pool_parallel_vec(
        l.lattice().get_points(),
        l,
        &|point, l| integrate_efield(point, l, delta_t),
        number_of_thread,
        l.lattice().get_number_of_points(),
        l.lattice(),
        Vector3::from_element(Su3Adjoint::default()),
    )
}

/// Basic symplectic Euler integrator
#[derive(Debug, PartialEq, Clone, Copy, Hash)]
pub struct SymplecticEuler
{
    number_of_thread: usize,
}

impl SymplecticEuler {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
}

impl Default for SymplecticEuler {
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State> SymplecticIntegrator<State, State> for SymplecticEuler
    where State: LatticeHamiltonianSimulationState + LatticeHamiltonianSimulationStateNew
{}


impl<State> Integrator<State, State> for  SymplecticEuler
    where State: LatticeHamiltonianSimulationState + LatticeHamiltonianSimulationStateNew
{
    fn integrate(&self, l: &State, delta_t: Real) ->  Result<State, SimulationError> {
        let number_of_thread = self.number_of_thread;
        let link_matrix = get_link_matrix_integrate(l, number_of_thread, delta_t)?;
        let e_field = get_e_field_integrate(l, number_of_thread, delta_t)?;
        
        State::new(l.lattice().clone(), l.beta(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
}

/// Basic symplectic Euler integrator used for switching between syn and leapfrog
#[derive(Debug, PartialEq, Clone, Copy, Hash)]
pub struct SymplecticEulerToLeap
{
    number_of_thread: usize,
}

impl SymplecticEulerToLeap {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
}

impl Default for SymplecticEulerToLeap{
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State1, State2> SymplecticIntegrator<State1, State2> for SymplecticEulerToLeap
    where State1: LatticeHamiltonianSimulationState + SimulationStateSynchrone + LatticeHamiltonianSimulationStateNew,
    State2: LatticeHamiltonianSimulationState + SimulationStateLeapFrog + LatticeHamiltonianSimulationStateNew
{}

impl<State1, State2> Integrator<State1, State2> for  SymplecticEulerToLeap
    where State1: LatticeHamiltonianSimulationState + SimulationStateSynchrone + LatticeHamiltonianSimulationStateNew,
    State2: LatticeHamiltonianSimulationState + SimulationStateLeapFrog + LatticeHamiltonianSimulationStateNew
{
    fn integrate(&self, l: &State1, delta_t: Real) ->  Result<State2, SimulationError> {
        let number_of_thread = self.number_of_thread;
        let e_field = get_e_field_integrate(l, number_of_thread, delta_t/ 2_f64)?;
        // we do not advance the step counter
        State2::new(l.lattice().clone(), l.beta(), EField::new(e_field), l.link_matrix().clone(), l.t())
    }
}

/// Basic symplectic Euler integrator used for switching between syn and leapfrog
#[derive(Debug, PartialEq, Clone, Copy, Hash)]
pub struct SymplecticEulerToSynch
{
    number_of_thread: usize,
}

impl SymplecticEulerToSynch {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
}

impl Default for SymplecticEulerToSynch{
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State1, State2> SymplecticIntegrator<State1, State2> for SymplecticEulerToSynch
    where State1: LatticeHamiltonianSimulationState + SimulationStateLeapFrog + LatticeHamiltonianSimulationStateNew,
    State2: LatticeHamiltonianSimulationState + SimulationStateSynchrone + LatticeHamiltonianSimulationStateNew
{}

impl<State1, State2> Integrator<State1, State2> for  SymplecticEulerToSynch
    where State1: LatticeHamiltonianSimulationState + SimulationStateLeapFrog + LatticeHamiltonianSimulationStateNew,
    State2: LatticeHamiltonianSimulationState + SimulationStateSynchrone + LatticeHamiltonianSimulationStateNew
{
    fn integrate(&self, l: &State1, delta_t: Real) ->  Result<State2, SimulationError> {
        let number_of_thread = self.number_of_thread;
        let link_matrix = get_link_matrix_integrate(l, number_of_thread, delta_t)?;
        let e_field = get_e_field_integrate(l, number_of_thread, delta_t/ 2_f64)?;
        // we do not advance the step counter
        State2::new(l.lattice().clone(), l.beta(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
}
