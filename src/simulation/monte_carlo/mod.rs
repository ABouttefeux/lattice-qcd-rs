//! Module for Monte-Carlo algrorithme, see the trait [`MonteCarlo`].
//!
//! This is one of the way to carry out simulation. This work by taking a state and progressively changing it (most of the time randomly).
//!
//! # Examples
//! see [`MetropolisHastingsSweep`], [`HeatBathSweep`], [`overrelaxation`] etc...

use std::marker::PhantomData;

use na::ComplexField;
use rand_distr::Distribution;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        field::LinkMatrix,
        lattice::{Direction, LatticeCyclique, LatticeLink, LatticeLinkCanonical},
        Complex, Real,
    },
    state::{LatticeState, LatticeStateDefault},
};

pub mod heat_bath;
pub mod hybrid;
pub mod hybrid_monte_carlo;
pub mod metropolis_hastings;
pub mod metropolis_hastings_sweep;
pub mod overrelaxation;

pub use heat_bath::*;
pub use hybrid::*;
pub use hybrid_monte_carlo::*;
pub use metropolis_hastings::*;
pub use metropolis_hastings_sweep::*;
pub use overrelaxation::*;

/// Monte-Carlo algorithm, giving the next element in the simulation.
/// It is also a Markov chain.
///
/// # Example
/// ```
/// # use std::error::Error;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// use lattice_qcd_rs::error::ImplementationError;
/// use lattice_qcd_rs::simulation::{
///     LatticeState, LatticeStateDefault, MetropolisHastingsSweep, MonteCarlo,
/// };
/// use rand::SeedableRng;
///
/// let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
/// let mut mh = MetropolisHastingsSweep::new(1, 0.1_f64, rng)
///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// // Realistically you want more steps than 10
///
/// let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 6_f64, 4)?;
/// for _ in 0..10 {
///     state = mh.get_next_element(state)?;
///     // or state.monte_carlo_step(&mut hmc)?;
///     // operation to track the progress or the evolution
/// }
/// // operation at the end of the simulation
/// #     Ok(())
/// # }
/// ```
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

    /// Generate a radom element from the previous element (like a Markov chain).
    ///
    /// # Errors
    /// Gives an error if a potential next ellement cannot be generated.
    fn get_potential_next_element<Rng>(
        &mut self,
        state: &State,
        rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized;

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
    fn get_next_element_default<Rng>(
        &mut self,
        state: State,
        rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
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

/// A wrapper used to implement [`MonteCarlo`] from a [`MonteCarloDefault`]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
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
    getter!(
        /// Get a ref to the rng.
        pub,
        rng,
        Rng
    );

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

    /// Get a mutable reference to the rng
    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }
}

impl<MCD, State, Rng, const D: usize> AsRef<Rng> for McWrapper<MCD, State, Rng, D>
where
    MCD: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
{
    fn as_ref(&self) -> &Rng {
        self.rng()
    }
}

impl<MCD, State, Rng, const D: usize> AsMut<Rng> for McWrapper<MCD, State, Rng, D>
where
    MCD: MonteCarloDefault<State, D>,
    State: LatticeState<D>,
    Rng: rand::Rng,
{
    fn as_mut(&mut self) -> &mut Rng {
        self.rng_mut()
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

impl<MCD, State, Rng, const D: usize> Default for McWrapper<MCD, State, Rng, D>
where
    MCD: MonteCarloDefault<State, D> + Default,
    State: LatticeState<D>,
    Rng: rand::Rng + Default,
{
    fn default() -> Self {
        Self::new(MCD::default(), Rng::default())
    }
}

impl<MCD, State, Rng, const D: usize> std::fmt::Display for McWrapper<MCD, State, Rng, D>
where
    MCD: MonteCarloDefault<State, D> + std::fmt::Display,
    State: LatticeState<D>,
    Rng: rand::Rng + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Monte Carlo wrapper method {} with rng {}",
            self.mcd(),
            self.rng(),
        )
    }
}

/// Get the delta of energy by changing a link.
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
/// return the staple
#[inline]
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
