
//! Hybrid Monte Carlo methode

use super::{
    MonteCarlo,
    MonteCarloDefault,
    super::{
        super::{
            Real,
            integrator::SymplecticIntegrator,
            field::Su3Adjoint,
            lattice::{
                Direction,
                DirectionList,
            }
        },
        state::{
            SimulationStateSynchrone,
            SimulationStateLeap,
            LatticeState,
            LatticeHamiltonianSimulationStateSyncDefault,
        },
        SimulationError,
    },
};
use std::marker::PhantomData;
use rand_distr::Distribution;
use na::{
    DimName,
    DefaultAllocator,
    VectorN,
    base::allocator::Allocator,
};

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

/// Hybrid Monte Carlo algorithm ( HCM for short).
///
/// The idea of HCM is to generate a random set on conjugate momenta to the link matrices.
/// This conjugatewd momenta is also refed as the "Electric" field
/// or `e_field` with distribution N(0, 1 / beta). And to solve the equation of motion.
/// The new state is accepted with probability Exp( -H_old + H_new) where the Hamiltonian has an extra term Tr(E_i ^ 2).
/// The advantage is that the simulation can be done in a simpleptic way i.e. it conserved the Hamiltonian.
/// Which means that the methode has a high acceptance rate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HybridMonteCarlo<State, Rng, I, D>
    where State: LatticeState<D> + Clone,
    LatticeHamiltonianSimulationStateSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State, D>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State, D>, D>, D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    internal: HybridMonteCarloInternal<LatticeHamiltonianSimulationStateSyncDefault<State, D>, I, D>,
    rng: Rng,
}

impl<State, Rng, I, D> HybridMonteCarlo<State, Rng, I, D>
    where State: LatticeState<D> + Clone,
    LatticeHamiltonianSimulationStateSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State, D>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State, D>, D>, D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
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
            internal: HybridMonteCarloInternal::<LatticeHamiltonianSimulationStateSyncDefault<State, D>, I, D>::new(delta_t, number_of_steps, integrator),
            rng,
        }
    }
    
    /// Get a mutlable reference to the rng.
    pub fn get_rng(&mut self) -> &mut Rng{
        &mut self.rng
    }
    
    /// Get the last probably of acceptance of the random change.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
}

impl<State, Rng, I, D> MonteCarlo<State, D> for HybridMonteCarlo<State, Rng, I, D>
    where State: LatticeState<D> + Clone,
    LatticeHamiltonianSimulationStateSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State, D>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State ,D> ,D>, D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State, D>::new_random_e_state(state, self.get_rng());
        self.internal.get_next_element_default(state_internal, &mut self.rng).map(|el| el.get_state_owned())
    }
}

/// internal structure for HybridMonteCarlo using [`LatticeHamiltonianSimulationState`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
struct HybridMonteCarloInternal<State, I, D>
    where State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator: I,
    #[cfg_attr(feature = "serde-serialize", serde(skip) )]
    _phantom: PhantomData<(State, D)>,
}

impl<State, I, D> HybridMonteCarloInternal<State, I, D>
    where State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
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

impl<State, I, D> MonteCarloDefault<State, D> for HybridMonteCarloInternal<State, I, D>
    where State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    
    fn get_potential_next_element(&mut self, state: &State, _rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        state.simulate_using_leapfrog_n_auto(self.delta_t, self.number_of_steps, &self.integrator)
    }
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (old_state.get_hamiltonian_total() - new_state.get_hamiltonian_total()).exp()
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
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HybridMonteCarloDiagnostic<State, Rng, I, D>
    where State: LatticeState<D> + Clone,
    LatticeHamiltonianSimulationStateSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State, D>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State, D>, D>, D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    internal: HybridMonteCarloInternalDiagnostics<LatticeHamiltonianSimulationStateSyncDefault<State, D>, I, D>,
    rng: Rng,
}

impl<State, Rng, I, D> HybridMonteCarloDiagnostic<State, Rng, I, D>
    where State: LatticeState<D> + Clone,
    LatticeHamiltonianSimulationStateSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State, D>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State, D>, D>, D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
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
            internal: HybridMonteCarloInternalDiagnostics::<LatticeHamiltonianSimulationStateSyncDefault<State, D>, I, D>::new(delta_t, number_of_steps, integrator),
            rng,
        }
    }
    
    /// Get a mutlable reference to the rng.
    pub fn get_rng(&mut self) -> &mut Rng{
        &mut self.rng
    }
    
    /// Get the last probably of acceptance of the random change.
    pub fn prob_replace_last(&self) -> Real {
        self.internal.prob_replace_last()
    }
    
    /// Get if last step has accepted the replacement.
    pub fn has_replace_last(&self) -> bool {
        self.internal.has_replace_last()
    }
    
    /// Get the last probably of acceptance of the random change.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
}

impl<State, Rng, I, D> MonteCarlo<State, D> for HybridMonteCarloDiagnostic<State, Rng, I, D>
    where State: LatticeState<D> + Clone,
    LatticeHamiltonianSimulationStateSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State, D>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State, D>, D>, D>,
    Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State, D>::new_random_e_state(state, self.get_rng());
        self.internal.get_next_element_default(state_internal, &mut self.rng).map(|el| el.get_state_owned())
    }
}

/// internal structure for HybridMonteCarlo using [`LatticeHamiltonianSimulationState`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
struct HybridMonteCarloInternalDiagnostics<State, I, D>
    where State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator: I,
    has_replace_last: bool,
    prob_replace_last: Real,
    #[cfg_attr(feature = "serde-serialize", serde(skip) )]
    _phantom: PhantomData<(State, D)>,
}

impl<State, I, D> HybridMonteCarloInternalDiagnostics<State, I, D>
    where State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
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
    
    /// Get the last probably of acceptance of the random change.
    pub fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }
    
    /// Get if last step has accepted the replacement.
    pub fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
}

impl<State, I, D> MonteCarloDefault<State, D> for HybridMonteCarloInternalDiagnostics<State, I, D>
    where State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    
    fn get_potential_next_element(&mut self, state: &State, _rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        state.simulate_using_leapfrog_n_auto(self.delta_t, self.number_of_steps, &self.integrator)
    }
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (old_state.get_hamiltonian_total() - new_state.get_hamiltonian_total()).exp()
            .min(1_f64)
            .max(0_f64)
    }
    
    fn get_next_element_default(&mut self, state: State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next).min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            self.has_replace_last = true;
            return Ok(potential_next);
        }
        else{
            self.has_replace_last = false;
            return Ok(state);
        }
    }
    
}
