
//! Basic symplectic Euler integrator
//!
//! See [`SymplecticEuler`]

use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
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
            LatticeState,
            LatticeHamiltonianSimulationStateNew,
            SimulationStateLeap,
        },
        lattice::{
            LatticeCyclique,
            Direction,
            DirectionList,
        },
    },
    SymplecticIntegrator,
    integrate_link,
    integrate_efield,
};
use std::vec::Vec;
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

/// Basic symplectic Euler integrator
///
/// slightly slower than [`super::SymplecticEulerRayon`] (for aproriate choice of `number_of_thread`)
/// but use less memory
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SymplecticEuler
{
    number_of_thread: usize,
}

impl SymplecticEuler {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
    
    getter_copy!(const,
        /// getter on the number of thread the integrator use.
        number_of_thread, usize
    );
    
    fn get_link_matrix_integrate<State, D> (self, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>, delta_t: Real) -> Result<Vec<CMatrix3>, ThreadError>
        where State: LatticeHamiltonianSimulationState<D>,
        D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Sync + Send,
        DefaultAllocator: Allocator<Su3Adjoint, D>,
        VectorN<Su3Adjoint, D>: Sync + Send,
        D: Eq,
        Direction<D>: DirectionList,
    {
        run_pool_parallel_vec(
            lattice.get_links(),
            &(link_matrix, e_field, lattice),
            &|link, (link_matrix, e_field, lattice)| integrate_link::<State, D>(link, link_matrix, e_field, lattice, delta_t),
            self.number_of_thread,
            lattice.get_number_of_canonical_links_space(),
            lattice,
            &CMatrix3::zeros(),
        )
    }
    
    fn get_e_field_integrate<State, D> (self, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>, delta_t: Real) -> Result<Vec<VectorN<Su3Adjoint, D>>, ThreadError>
        where State: LatticeHamiltonianSimulationState<D>,
        D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Sync + Send,
        DefaultAllocator: Allocator<Su3Adjoint, D>,
        VectorN<Su3Adjoint, D>: Send + Sync,
        D: Eq,
        Direction<D>: DirectionList,
    {
        run_pool_parallel_vec(
            lattice.get_points(),
            &(link_matrix, e_field, lattice),
            &|point, (link_matrix, e_field, lattice)| integrate_efield::<State, D>(point, link_matrix, e_field, lattice, delta_t),
            self.number_of_thread,
            lattice.get_number_of_points(),
            lattice,
            &VectorN::<_, D>::from_element(Su3Adjoint::default()),
        )
    }
}

impl Default for SymplecticEuler {
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State, D> SymplecticIntegrator<State, SimulationStateLeap<State, D>, D> for SymplecticEuler
    where State: SimulationStateSynchrone<D> + LatticeHamiltonianSimulationState<D> + LatticeHamiltonianSimulationStateNew<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Sync + Send,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Send + Sync,
    D: Eq,
    Direction<D>: DirectionList,
{
    fn integrate_sync_sync(&self, l: &State, delta_t: Real) -> Result<State, SimulationError> {
        let link_matrix = self.get_link_matrix_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t)?;
        let e_field = self.get_e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t)?;
        
        State::new(l.lattice().clone(), l.beta(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
    
    fn integrate_leap_leap(&self, l: &SimulationStateLeap<State, D>, delta_t: Real) -> Result<SimulationStateLeap<State, D>, SimulationError> {
        let link_matrix = LinkMatrix::new(self.get_link_matrix_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t)?);
        let e_field = EField::new(self.get_e_field_integrate::<State, D>(&link_matrix, l.e_field(), l.lattice(), delta_t)?);
        SimulationStateLeap::<State, D>::new(l.lattice().clone(), l.beta(), e_field, link_matrix, l.t() + 1)
    }
    
    fn integrate_sync_leap(&self, l: &State, delta_t: Real) -> Result<SimulationStateLeap<State, D>, SimulationError> {
        let e_field = self.get_e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t / 2_f64)?;
        // we do not advance the step counter
        SimulationStateLeap::<State, D>::new(l.lattice().clone(), l.beta(), EField::new(e_field), l.link_matrix().clone(), l.t())
    }
    
    fn integrate_leap_sync(&self, l: &SimulationStateLeap<State, D>, delta_t: Real) -> Result<State, SimulationError>{
        let link_matrix = LinkMatrix::new(self.get_link_matrix_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t)?);
        // we advace the counter by one
        let e_field = EField::new(self.get_e_field_integrate::<State, D>(&link_matrix, l.e_field(), l.lattice(), delta_t / 2_f64)?);
        State::new(l.lattice().clone(), l.beta(), e_field, link_matrix, l.t() + 1)
    }
    
    fn integrate_symplectic(&self, l: &State, delta_t: Real) -> Result<State, SimulationError> {
        // override for optimization.
        // This remove a clone operation.
        
        let e_field_demi = EField::new(self.get_e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t / 2_f64)?);
        let link_matrix = LinkMatrix::new(self.get_link_matrix_integrate::<State, D>(l.link_matrix(), &e_field_demi, l.lattice(), delta_t)?);
        let e_field = EField::new(self.get_e_field_integrate::<State, D>(&link_matrix, &e_field_demi, l.lattice(), delta_t / 2_f64)?);
        
        State::new(l.lattice().clone(), l.beta(), e_field, link_matrix, l.t() + 1)
    }
}
