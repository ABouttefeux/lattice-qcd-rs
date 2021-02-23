
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
        lattice::LatticeCyclique,
    },
    SymplecticIntegrator,
    integrate_link,
    integrate_efield,
    CMatrix3,
};
use std::vec::Vec;
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
};

fn get_link_matrix_integrate<State, D> (link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>, delta_t: Real) -> Vec<CMatrix3>
    where State: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    D: Eq,
{
    run_pool_parallel_rayon(
        lattice.get_links(),
        &(link_matrix, e_field, lattice),
        |link, (link_matrix, e_field, lattice)| integrate_link::<State, D>(link, link_matrix, e_field, lattice, delta_t),
    )
}

fn get_e_field_integrate<State, D> (link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>, delta_t: Real) -> Vec<VectorN<Su3Adjoint, D>>
    where State: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    D: Eq,
{
    run_pool_parallel_rayon(
        lattice.get_points(),
        &(link_matrix, e_field, lattice),
        |point, (link_matrix, e_field, lattice)| integrate_efield::<State, D>(point, link_matrix, e_field, lattice, delta_t),
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

impl<State, D> SymplecticIntegrator<State, SimulationStateLeap<State, D>, D> for SymplecticEulerRayon
    where State: SimulationStateSynchrone<D> + LatticeHamiltonianSimulationState<D> + LatticeHamiltonianSimulationStateNew<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    D: Eq,
{
    fn integrate_sync_sync(&self, l: &State, delta_t: Real) -> Result<State, SimulationError> {
        let link_matrix = get_link_matrix_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t);
        let e_field = get_e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t);
        
        State::new(l.lattice().clone(), l.beta(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
    
    fn integrate_leap_leap(&self, l: &SimulationStateLeap<State, D>, delta_t: Real) -> Result<SimulationStateLeap<State, D>, SimulationError> {
        let link_matrix = LinkMatrix::new(get_link_matrix_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t));
        
        let e_field = EField::new(get_e_field_integrate::<State, D>(&link_matrix, l.e_field(), l.lattice(), delta_t));
        SimulationStateLeap::<State, D>::new(l.lattice().clone(), l.beta(), e_field, link_matrix, l.t() + 1)
    }
    
    fn integrate_sync_leap(&self, l: &State, delta_t: Real) -> Result<SimulationStateLeap<State, D>, SimulationError> {
        let e_field = get_e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t / 2_f64);
        
        // we do not advace the time counter
        SimulationStateLeap::<State, D>::new(l.lattice().clone(), l.beta(), EField::new(e_field), l.link_matrix().clone(), l.t())
    }
    
    fn integrate_leap_sync(&self, l: &SimulationStateLeap<State, D>, delta_t: Real) -> Result<State, SimulationError>{
        // TODO correct
        let link_matrix = LinkMatrix::new(get_link_matrix_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t));
        // we advace the counter by one
        // I do not like the clone of e_field :(.
        let e_field = EField::new(get_e_field_integrate::<State, D>(&link_matrix, l.e_field(), l.lattice(), delta_t / 2_f64));
        State::new(l.lattice().clone(), l.beta(), e_field, link_matrix, l.t() + 1)
    }
}
