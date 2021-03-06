//! Module for Monte-Carlo algrorithme and the trait [`MonteCarlo`]

use std::marker::PhantomData;

use na::ComplexField;
use rand_distr::Distribution;

use super::{
    super::{
        field::LinkMatrix,
        lattice::{Direction, LatticeCyclique, LatticeLink, LatticeLinkCanonical},
        Complex, Real,
    },
    state::{LatticeState, LatticeStateDefault},
};

pub mod heat_bath;
pub mod hybride;
pub mod hybride_monte_carlo;
pub mod metropolis_hastings;
pub mod metropolis_hastings_sweep;
pub mod overrelaxation;

pub use heat_bath::*;
pub use hybride::*;
pub use hybride_monte_carlo::*;
pub use metropolis_hastings::*;
pub use metropolis_hastings_sweep::*;
pub use overrelaxation::*;

/// Monte-Carlo algorithm, giving the next element in the simulation.
/// It is also a Markov chain
pub trait MonteCarlo<State, const D: usize>
where
    State: LatticeState<D>,
{
    /// Error returned while getting the next ellement.
    type Error;

    /// Do one Monte Carlo simulation step.
    ///
    /// # Errors
    /// Return an error if the simulation failed
    fn get_next_element(&mut self, state: State) -> Result<State, Self::Error>;
}

/// Some times is is esayer to just implement a potential next element, the rest is done automatically.
///
/// To get an [`MonteCarlo`] use the wrapper [`McWrapper`]
pub trait MonteCarloDefault<State, const D: usize>
where
    State: LatticeState<D>,
{
    /// Error returned while getting the next ellement.
    type Error;

    /// Generate a radom element from the previous element ( like a Markov chain).
    ///
    /// # Errors
    /// Gives an error if a potential next ellement cannot be generated.
    fn get_potential_next_element(
        &mut self,
        state: &State,
        rng: &mut impl rand::Rng,
    ) -> Result<State, Self::Error>;

    /// probability of the next element to replace the current one.
    ///
    /// by default it is Exp(-H_old) / Exp(-H_new).
    fn get_probability_of_replacement(old_state: &State, new_state: &State) -> Real {
        (old_state.get_hamiltonian_links() - new_state.get_hamiltonian_links())
            .exp()
            .min(1_f64)
            .max(0_f64)
    }

    /// Get the next element in the chain either the old state or a new one replacing it.
    ///
    /// # Errors
    /// Gives an error if a potential next ellement cannot be generated.
    fn get_next_element_default(
        &mut self,
        state: State,
        rng: &mut impl rand::Rng,
    ) -> Result<State, Self::Error> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next)
            .min(1_f64)
            .max(0_f64);
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            Ok(potential_next)
        }
        else {
            Ok(state)
        }
    }
}

/// A arapper used to implement [`MonteCarlo`] from a [`MonteCarloDefault`]
#[derive(Clone, Debug)]
pub struct McWrapper<MCD, State, Rng, const D: usize>
where
    MCD: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
{
    mcd: MCD,
    rng: Rng,
    _phantom: PhantomData<State>,
}

impl<MCD, State, Rng, const D: usize> McWrapper<MCD, State, Rng, D>
where
    MCD: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
{
    /// Create the wrapper.
    pub fn new(mcd: MCD, rng: Rng) -> Self {
        Self {
            mcd,
            rng,
            _phantom: PhantomData,
        }
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

impl<T, State, Rng, const D: usize> MonteCarlo<State, D> for McWrapper<T, State, Rng, D>
where
    T: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
{
    type Error = T::Error;

    fn get_next_element(&mut self, state: State) -> Result<State, Self::Error> {
        self.mcd.get_next_element_default(state, &mut self.rng)
    }
}

#[inline]
fn get_delta_s_old_new_cmp<const D: usize>(
    link_matrix: &LinkMatrix,
    lattice: &LatticeCyclique<D>,
    link: &LatticeLinkCanonical<D>,
    new_link: &na::Matrix3<Complex>,
    beta: Real,
    old_matrix: &na::Matrix3<Complex>,
) -> Real {
    let a = get_staple(link_matrix, lattice, link);
    -((new_link - old_matrix) * a).trace().real() * beta / LatticeStateDefault::<D>::CA
}

// TODO move in state
fn get_staple<const D: usize>(
    link_matrix: &LinkMatrix,
    lattice: &LatticeCyclique<D>,
    link: &LatticeLinkCanonical<D>,
) -> na::Matrix3<Complex> {
    let dir_j = link.dir();
    Direction::<D>::positive_directions()
        .iter()
        .filter(|dir_i| *dir_i != dir_j)
        .map(|dir_i| {
            let el_1 = link_matrix
                .get_sij(link.pos(), dir_j, &dir_i, lattice)
                .unwrap()
                .adjoint();
            let l_1 = LatticeLink::new(lattice.add_point_direction(*link.pos(), dir_j), -dir_i);
            let u1 = link_matrix.get_matrix(&l_1, lattice).unwrap();
            let l_2 = LatticeLink::new(lattice.add_point_direction(*link.pos(), &-dir_i), *dir_j);
            let u2 = link_matrix.get_matrix(&l_2, lattice).unwrap().adjoint();
            let l_3 = LatticeLink::new(lattice.add_point_direction(*link.pos(), &-dir_i), *dir_i);
            let u3 = link_matrix.get_matrix(&l_3, lattice).unwrap();
            el_1 + u1 * u2 * u3
        })
        .sum()
}
