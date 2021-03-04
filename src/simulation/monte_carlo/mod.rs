
//! Module for Monte-Carlo algrorithme and the trait [`MonteCarlo`]

use super::{
    super::{
        Real,
        Complex,
        lattice::{
            Direction,
            DirectionList,
            LatticeCyclique,
            LatticeLinkCanonical,
            LatticeLink,
        },
        field::LinkMatrix,
    },
    state::{
        LatticeState,
        LatticeStateDefault,
    },
    SimulationError,
};
use std::marker::PhantomData;
use rand_distr::Distribution;
use na::{
    DimName,
    DefaultAllocator,
    VectorN,
    base::allocator::Allocator,
    ComplexField,
};

pub mod hybride_monte_carlo;
pub mod metropolis_hastings;
pub mod metropolis_hastings_sweep;
pub mod heat_bath;
pub mod overrelaxation;
pub mod hybride;

pub use hybride_monte_carlo::*;
pub use metropolis_hastings::*;
pub use metropolis_hastings_sweep::*;
pub use heat_bath::*;
pub use overrelaxation::*;
pub use hybride::*;


/// Monte-Carlo algorithm, giving the next element in the simulation.
/// It is also a Markov chain
pub trait MonteCarlo<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError>;
}

/// Some times is is esayer to just implement a potential next element, the rest is done automatically.
///
/// To get an [`MonteCarlo`] use the wrapper [`MCWrapper`]
pub trait MonteCarloDefault<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    
    /// Generate a radom element from the previous element ( like a Markov chain).
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError>;
    
    /// probability of the next element to replace the current one.
    ///
    /// by default it is Exp(-H_old) / Exp(-H_new).
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (old_state.get_hamiltonian_links() - new_state.get_hamiltonian_links()).exp()
            .min(1_f64)
            .max(0_f64)
    }
    
    /// Get the next element in the chain either the old state or a new one replacing it.
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next).min(1_f64).max(0_f64);
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            return Ok(potential_next);
        }
        else{
            return Ok(state);
        }
    }
}

/// A arapper used to implement [`MonteCarlo`] from a [`MonteCarloDefault`]
#[derive(Clone, Debug)]
pub struct MCWrapper<MCD, State, D, Rng>
    where MCD: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    mcd: MCD,
    rng: Rng,
    _phantom: PhantomData<(State, D)>,
}

impl<MCD, State, Rng, D> MCWrapper<MCD, State, D, Rng>
    where MCD: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// Create the wrapper.
    pub fn new(mcd: MCD, rng: Rng) -> Self{
        Self{mcd, rng, _phantom: PhantomData}
    }
    
    /// deconstruct the structure to get back the rng if necessary
    pub fn deconstruct(self) -> (MCD, Rng) {
        (self.mcd, self.rng)
    }
    
    /// Get a reference to the [`MonteCarloDefault`] inside the wrapper.
    pub fn mcd(&self) -> &MCD {
        &self.mcd
    }
}

impl<T, State, D, Rng> MonteCarlo<State, D> for MCWrapper<T, State, D, Rng>
    where T: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        self.mcd.get_next_element_default(state, &mut self.rng)
    }
}

#[inline]
fn get_delta_s_old_new_cmp<D>(
    link_matrix: &LinkMatrix,
    lattice: &LatticeCyclique<D>,
    link: &LatticeLinkCanonical<D>,
    new_link: &na::Matrix3<Complex>,
    beta : Real,
    old_matrix: &na::Matrix3<Complex>,
) -> Real
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
    DefaultAllocator: Allocator<na::Complex<Real>, na::U3, na::U3>,
{
    let a = get_straple(link_matrix, lattice, link);
    -((new_link - old_matrix) * a).trace().real() * beta / LatticeStateDefault::CA
}


fn get_straple<D>(
    link_matrix: &LinkMatrix,
    lattice: &LatticeCyclique<D>,
    link: &LatticeLinkCanonical<D>,
) -> na::Matrix3<Complex>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
    DefaultAllocator: Allocator<na::Complex<Real>, na::U3, na::U3>,
{
    let dir_j = link.dir();
    Direction::<D>::get_all_positive_directions().iter()
        .filter(|dir_i| *dir_i != dir_j ).map(|dir_i| {
            let el_1 = link_matrix.get_sij(link.pos(), dir_j, &dir_i, lattice).unwrap().adjoint();
            let l_1 = LatticeLink::new(lattice.add_point_direction(*link.pos(), dir_j), - dir_i);
            let u1 = link_matrix.get_matrix(&l_1, lattice).unwrap();
            let l_2 = LatticeLink::new(lattice.add_point_direction(*link.pos(), &- dir_i), *dir_j);
            let u2 = link_matrix.get_matrix(&l_2, lattice).unwrap().adjoint();
            let l_3 = LatticeLink::new(lattice.add_point_direction(*link.pos(), &- dir_i), *dir_i);
            let u3 = link_matrix.get_matrix(&l_3, lattice).unwrap();
            el_1 + u1 * u2 * u3
        }).sum()
}
