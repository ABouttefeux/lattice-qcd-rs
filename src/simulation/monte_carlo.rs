
//! Module for Monte-Carlo algrorithme and the trait [`MonteCarlo`]

use super::{
    super::{
        Real,
        integrator::SymplecticIntegrator,
        field::{
            LinkMatrix,
        },
        su3,
    },
    state::{
        SimulationStateSynchrone,
        SimulationStateLeap,
        LatticeState,
        LatticeHamiltonianSimulationStateSyncDefault,
        LatticeStateNew,
    },
    SimulationError,
};
use std::marker::PhantomData;
use rand_distr::Distribution;

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

/// Hybrid Monte Carlo algorithm ( HCM for short).
///
/// The idea of HCM is to generate a random set on conjugate momenta to the link matrices.
/// This conjugatewd momenta is also refed as the "Electric" field
/// or `e_field` with distribution N(0, 1 / beta). And to solve the equation of motion.
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
        //let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_e_cold(state);
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
        ( old_state.get_hamiltonian_total() - new_state.get_hamiltonian_total()).exp()
            .min(1_f64)
            .max(0_f64)
    }
    
}

/// Hybrid Monte Carlo algorithm ( HCM for short) with diagnostics.
///
/// The idea of HCM is to generate a random set on conjugate momenta to the link matrices.
/// This conjugatewd momenta is also refed as the "Electric" field
/// or `e_field` with distribution N(0, 1 / beta). And to solve the equation of motion.
/// The new state is accepted with probability Exp( -H_old + H_new) where the Hamiltonian has an extra term Tr(E_i ^ 2).
/// The advantage is that the simulation can be done in a simpleptic way i.e. it conserved the Hamiltonian.
/// Which means that the methode has a high acceptance rate.
#[derive(Clone, Debug, PartialEq)]
pub struct HybridMonteCarloDiagnostic<State, Rng, I>
    where State: LatticeState + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    internal: HybridMonteCarloInternalDiagnostics<LatticeHamiltonianSimulationStateSyncDefault<State>, I>,
    rng: Rng,
}

impl<State, Rng, I> HybridMonteCarloDiagnostic<State, Rng, I>
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
            internal: HybridMonteCarloInternalDiagnostics::<LatticeHamiltonianSimulationStateSyncDefault<State>, I>::new(delta_t, number_of_steps, integrator),
            rng,
        }
    }
    
    pub fn get_rng(&mut self) -> &mut Rng{
        &mut self.rng
    }
    
    pub fn prob_replace_last(&self) -> Real {
        self.internal.prob_replace_last()
    }
    
    pub fn has_replace_last(&self) -> bool {
        self.internal.has_replace_last()
    }
}

impl<State, Rng, I> MonteCarlo<State> for HybridMonteCarloDiagnostic<State, Rng, I>
    where State: LatticeState + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_random_e_state(state, self.get_rng());
        //let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_e_cold(state);
        self.internal.get_next_element_default(state_internal, &mut self.rng).map(|el| el.get_state_owned())
    }
}

/// internal structure for HybridMonteCarlo using [`LatticeHamiltonianSimulationState`]
#[derive(Clone, Debug, PartialEq)]
struct HybridMonteCarloInternalDiagnostics<State, I>
    where State: SimulationStateSynchrone,
    I: SymplecticIntegrator<State, SimulationStateLeap<State>>,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator: I,
    has_replace_last: bool,
    prob_replace_last: Real,
    _phantom: PhantomData<State>,
}

impl<State, I> HybridMonteCarloInternalDiagnostics<State, I>
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
            has_replace_last: false,
            prob_replace_last: 0_f64,
            _phantom: PhantomData,
        }
    }
    
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    pub fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
}

impl<State, I> MonteCarloDefault<State> for HybridMonteCarloInternalDiagnostics<State, I>
    where State: SimulationStateSynchrone,
    I: SymplecticIntegrator<State, SimulationStateLeap<State>>,
{
    
    fn get_potential_next_element(&mut self, state: &State, _rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        state.simulate_using_leapfrog_n_auto(self.delta_t, self.number_of_steps, &self.integrator)
    }
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        ( old_state.get_hamiltonian_total() - new_state.get_hamiltonian_total()).exp()
            .min(1_f64)
            .max(0_f64)
    }
    
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next).min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng)  {
            self.has_replace_last = true;
            return Ok(potential_next);
        }
        else{
            self.has_replace_last = false;
            return Ok(state);
        }
    }
    
}

/// Metropolis Hastings algorithm.
pub struct MetropolisHastings<State>
    where State: LatticeState,
{
    number_of_update: usize,
    spread: Real,
    _phantom: PhantomData<State>,
}

impl<State> MetropolisHastings<State>
    where State: LatticeState,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            _phantom: PhantomData,
        })
    }
}

impl<State> MonteCarloDefault<State> for MetropolisHastings<State>
    where State: LatticeState + LatticeStateNew,
{
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
        });
        State::new(state.lattice().clone(), state.beta(), LinkMatrix::new(link_matrix))
    }
}

/// Metropolis Hastings algorithm with diagnostics.
pub struct MetropolisHastingsDiagnostic<State>
    where State: LatticeState,
{
    number_of_update: usize,
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
    _phantom: PhantomData<State>,
}

impl<State> MetropolisHastingsDiagnostic<State>
    where State: LatticeState,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            has_replace_last: false,
            prob_replace_last: 0_f64,
            _phantom: PhantomData,
        })
    }
    
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    pub fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
}

impl<State> MonteCarloDefault<State> for MetropolisHastingsDiagnostic<State>
    where State: LatticeState + LatticeStateNew,
{
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
        });
        State::new(state.lattice().clone(), state.beta(), LinkMatrix::new(link_matrix))
    }
    
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next).min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng)  {
            self.has_replace_last = true;
            return Ok(potential_next);
        }
        else{
            self.has_replace_last = false;
            return Ok(state);
        }
    }
}
