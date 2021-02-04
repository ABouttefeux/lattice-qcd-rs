
//! Module for Monte-Carlo algrorithme and the trait [`MonteCarlo`]

use super::{
    super::{
        Real,
        integrator::SymplecticIntegrator,
    },
    state::{
        SimulationStateSynchrone,
        SimulationStateLeap,
        LatticeState,
        LatticeHamiltonianSimulationStateSyncDefault,
    },
    SimulationError,
};
use std::marker::PhantomData;
use rand_distr::Distribution;
use once_cell::sync::Lazy;

/// a Uniform distribution between [0, 1) (1 excluded)
static DISTRIBUTION_0_1: Lazy<rand::distributions::Uniform<Real>> = Lazy::new(
    || rand::distributions::Uniform::new(0_f64, 1_f64)
);

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
        (- old_state.get_hamiltonian_links() + new_state.get_hamiltonian_links()).exp().min(1_f64)
    }
    
    /// Get the next element in the chain either the old state or a new one replacing it.
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let r = DISTRIBUTION_0_1.sample(rng);
        if r < Self::get_probability_of_replacement(&state, &potential_next) {
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

/// Hybrid Monte Carlo algorithm ( HCM for short)
///
/// The idea of HCM is to generate a random set on conjugate momenta to the link matrices. This conjugatewd momenta is also refed as the "Electric" field
/// or `e_field` with distribution N(0, 1) (also called Standard Normal). And to solve the equation of motion.
/// The new state is accepted with probability Exp( -H_old + H_new) where the Hamiltonian has an extra term Tr(E_i ^ 2).
/// The advantage is that the simulation can be done in a simpleptic way i.e. it conserved the Hamiltonian.
/// Which means that the methode has a high acceptance rate.
#[derive(Clone, Debug, PartialEq)]
pub struct HybridMonteCarlo<State, Rng, I>
    where State: LatticeState + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    internal: HybridMonteCarloInternal<LatticeHamiltonianSimulationStateSyncDefault<State>, I>,
    rng: Rng,
}

impl<State, Rng, I> HybridMonteCarlo<State, Rng, I>
    where State: LatticeState + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    /// gvies the following parameter for the HCM :
    /// - delta_t is the step size per intgration of the equation of motion
    /// - number_of_steps is the number of time
    /// - integrator is the methode to solve the equation of motion
    /// - rng, a random number generator
    pub fn new(
        delta_t: Real,
        number_of_steps: usize,
        integrator: I,
        rng: Rng,
    ) -> Self {
        Self {
            internal: HybridMonteCarloInternal::<LatticeHamiltonianSimulationStateSyncDefault<State>, I>::new(delta_t, number_of_steps, integrator),
            rng,
        }
    }
    
    pub fn get_rng(&mut self) -> &mut Rng{
        &mut self.rng
    }
}

impl<State, Rng, I> MonteCarlo<State> for HybridMonteCarlo<State, Rng, I>
    where State: LatticeState + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_random_e_state(state, self.get_rng());
        self.internal.get_next_element_default(state_internal, &mut self.rng).map(|el| el.get_state_owned())
    }
}

/// internal structure for HybridMonteCarlo using [`LatticeHamiltonianSimulationState`]
#[derive(Clone, Debug, PartialEq)]
struct HybridMonteCarloInternal<State, I>
    where State: SimulationStateSynchrone,
    I: SymplecticIntegrator<State, SimulationStateLeap<State>>,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator: I,
    _phantom: PhantomData<State>,
}

impl<State, I> HybridMonteCarloInternal<State, I>
    where State: SimulationStateSynchrone,
    I: SymplecticIntegrator<State, SimulationStateLeap<State>>,
{
    /// see [HybridMonteCarlo::new]
    pub fn new(
        delta_t: Real,
        number_of_steps: usize,
        integrator: I,
    ) -> Self {
        Self {
            delta_t,
            number_of_steps,
            integrator,
            _phantom: PhantomData,
        }
    }
}

impl<State, I> MonteCarloDefault<State> for HybridMonteCarloInternal<State, I>
    where State: SimulationStateSynchrone,
    I: SymplecticIntegrator<State, SimulationStateLeap<State>>,
{
    
    fn get_potential_next_element(&mut self, state: &State, _rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        state.simulate_using_leapfrog_n_auto(self.delta_t, self.number_of_steps, &self.integrator)
    }
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (- old_state.get_hamiltonian_total() + new_state.get_hamiltonian_total()).exp().min(1_f64)
    }
    
}

/// Metropolis Hastings algorithm.
///
/// Not implmented yet. It will panic if you try calling [`MetropolisHastings::get_potential_next_element`]
struct MetropolisHastings<State>
    where State: LatticeState,
{
    _phantom: PhantomData<State>,
}

impl<State> MetropolisHastings<State>
    where State: LatticeState,
{
    pub fn new() -> Self {
        Self {_phantom: PhantomData}
    }
}

impl<State> Default for MetropolisHastings<State>
    where State: LatticeState,
{
    fn default() -> Self {
        Self::new()
    }
}


impl<State> MonteCarloDefault<State> for MetropolisHastings<State>
    where State: LatticeState,
{
    /// # Panic
    /// always panic becaus it is unimplemented yet
    fn get_potential_next_element(&mut self, _state: &State, _rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        todo!()
    }
}
