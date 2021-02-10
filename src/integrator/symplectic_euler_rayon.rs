
//! Basic symplectic Euler integrator using Rayon.
//!
//! See [`SymplecticEulerRayon`]
//!

use super::{
    super::{
        field::{
            EField,
            LinkMatrix,
            Su3Adjoint,
        },
        thread::{
            run_pool_parallel_rayon,
        },
        Real,
        simulation::{
            SimulationError,
            LatticeHamiltonianSimulationState,
            SimulationStateSynchrone,
            LatticeHamiltonianSimulationStateNew,
            SimulationStateLeap,
            LatticeState,
        },
    },
    SymplecticIntegrator,
    integrate_link,
    integrate_efield,
    CMatrix3,
};
use std::vec::Vec;
use na::Vector4;

fn get_link_matrix_integrate<State> (l: &State, delta_t: Real) -> Vec<CMatrix3>
    where State: LatticeHamiltonianSimulationState
{
    run_pool_parallel_rayon(
        l.lattice().get_links_space(),
        l,
        |link, l| integrate_link(link, l, delta_t),
    )
}

fn get_e_field_integrate<State> (l: &State, delta_t: Real) -> Vec<Vector4<Su3Adjoint>>
    where State: LatticeHamiltonianSimulationState
{
    run_pool_parallel_rayon(
        l.lattice().get_points(),
        l,
        |point, l| integrate_efield(point, l, delta_t),
    )
}

/// Basic symplectic Euler integrator using Rayon.
///
/// It is slightly faster than [`super::SymplecticEuler`] but use slightly more memory.
#[derive(Debug, PartialEq, Clone, Copy, Hash)]
pub struct SymplecticEulerRayon {}

impl SymplecticEulerRayon {
    /// Create a new SymplecticEulerRayon
    pub const fn new() -> Self {
        Self {}
    }
}

impl Default for SymplecticEulerRayon {
    /// Identical to [`SymplecticEulerRayon::new`].
    fn default() -> Self{
        Self::new()
    }
}

impl<State> SymplecticIntegrator<State, SimulationStateLeap<State>> for SymplecticEulerRayon
    where State: SimulationStateSynchrone + LatticeHamiltonianSimulationState + LatticeHamiltonianSimulationStateNew,
{
    fn integrate_sync_sync(&self, l: &State, delta_t: Real) ->  Result<State, SimulationError> {
        let link_matrix = get_link_matrix_integrate(l, delta_t);
        let e_field = get_e_field_integrate(l, delta_t);
        
        State::new(l.lattice().clone(), l.beta(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
    
    fn integrate_leap_leap(&self, l: &SimulationStateLeap<State>, delta_t: Real) ->  Result<SimulationStateLeap<State>, SimulationError> {
        let link_matrix = get_link_matrix_integrate(l, delta_t);
        // TODO I do not like the clone of e_field :(.
        // maybe try a structure which does not own e_field and link_matrix
        let mut state = SimulationStateLeap::<State>::new(l.lattice().clone(), l.beta(), l.e_field().clone(), LinkMatrix::new(link_matrix), l.t() + 1)?;
        let e_field = get_e_field_integrate(&state, delta_t);
        state.set_e_field(EField::new(e_field));
        Ok(state)
    }
    
    fn integrate_sync_leap(&self, l: &State, delta_t: Real) -> Result<SimulationStateLeap<State>, SimulationError> {
        let e_field = get_e_field_integrate(l, delta_t / 2_f64);
        
        // we do not advace the time counter
        SimulationStateLeap::<State>::new(l.lattice().clone(), l.beta(), EField::new(e_field), l.link_matrix().clone(), l.t())
    }
    
    fn integrate_leap_sync(&self, l: &SimulationStateLeap<State>, delta_t: Real) -> Result<State, SimulationError>{
        // TODO correct
        let link_matrix = get_link_matrix_integrate(l, delta_t);
        // we advace the counter by one
        // I do not like the clone of e_field :(.
        let mut state = State::new(l.lattice().clone(), l.beta(), l.e_field().clone(), LinkMatrix::new(link_matrix), l.t() + 1)?;
        let e_field = get_e_field_integrate(l, delta_t / 2_f64);
        state.set_e_field(EField::new(e_field));
        Ok(state)
    }
}
