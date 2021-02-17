
//! Module for Monte-Carlo algrorithme and the trait [`MonteCarlo`]

use super::{
    super::{
        Real,
    },
    state::{
        LatticeState,
    },
    SimulationError,
};
use std::marker::PhantomData;
use rand_distr::Distribution;

pub mod hybride_monte_carlo;
pub mod metropolis_hastings;

pub use hybride_monte_carlo::*;
pub use metropolis_hastings::*;


/// Monte-Carlo algorithm, giving the next element in the simulation.
/// It is also a Markov chain
pub trait MonteCarlo<State>
    where State: LatticeState,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError>;
}

/// Some times is is esayer to just implement a potential next element, the rest is done automatically.
///
/// To get an [`MonteCarlo`] use the wrapper [`MCWrapper`]
pub trait MonteCarloDefault<State>
    where State: LatticeState,
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
        if d.sample(rng)  {
            return Ok(potential_next);
        }
        else{
            return Ok(state);
        }
    }
}

/// A arapper used to implement [`MonteCarlo`] from a [`MonteCarloDefault`]
#[derive(Clone, Debug)]
pub struct MCWrapper<MCD, State, Rng>
    where MCD: MonteCarloDefault<State>,
    State: LatticeState,
    Rng: rand::Rng,
{
    mcd: MCD,
    rng: Rng,
    _phantom: PhantomData<State>,
}

impl<MCD, State, Rng> MCWrapper<MCD, State, Rng>
    where MCD: MonteCarloDefault<State>,
    State: LatticeState,
    Rng: rand::Rng,
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

impl<T, State, Rng> MonteCarlo<State> for MCWrapper<T, State, Rng>
    where T: MonteCarloDefault<State>,
    State: LatticeState,
    Rng: rand::Rng,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        self.mcd.get_next_element_default(state, &mut self.rng)
    }
}
