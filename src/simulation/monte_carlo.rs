
use super::{
    super::{
        Real,
        integrator::Integrator,
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
use once_cell::sync::Lazy;

static DISTRIBUTION_0_1: Lazy<rand::distributions::Uniform<Real>> = Lazy::new(
    || rand::distributions::Uniform::new(0_f64, 1_f64)
);

pub trait MonteCarlo<State>
    where State: LatticeState,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError>;
}

pub trait MonteCarloDefault<State>
    where State: LatticeState,
{
        
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError>;
        
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (- old_state.get_hamiltonian_links() + new_state.get_hamiltonian_links()).exp().min(1_f64)
    }
    
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
    pub fn new(mcd: MCD, rng: Rng) -> Self{
        Self{mcd, rng, _phantom: PhantomData}
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

#[derive(Clone, Debug, PartialEq)]
pub struct HybridMonteCarlo<State, Rng, ISTL, ILTL, ILTS>
    where State: LatticeStateNew + Clone,
    ISTL: Integrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTL: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTS: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, LatticeHamiltonianSimulationStateSyncDefault<State>> + Clone,
    Rng: rand::Rng,
{
    internal: HybridMonteCarloInternal<LatticeHamiltonianSimulationStateSyncDefault<State>, ISTL, ILTL, ILTS>,
    rng: Rng,
}

impl<State, Rng, ISTL, ILTL, ILTS> HybridMonteCarlo<State, Rng, ISTL, ILTL, ILTS>
    where State: LatticeStateNew + Clone,
    ISTL: Integrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTL: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTS: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, LatticeHamiltonianSimulationStateSyncDefault<State>> + Clone,
    Rng: rand::Rng,
{
    pub fn new(
        delta_t: Real,
        number_of_steps: usize,
        integrator_to_leap: ISTL,
        integrator_leap: ILTL,
        integrator_to_sync: ILTS,
        rng: Rng,
    ) -> Self {
        Self {
            internal: HybridMonteCarloInternal::<LatticeHamiltonianSimulationStateSyncDefault<State>, ISTL, ILTL, ILTS>::new(delta_t, number_of_steps, integrator_to_leap, integrator_leap, integrator_to_sync),
            rng,
        }
    }
    
    pub fn get_rng(&mut self) -> &mut Rng{
        &mut self.rng
    }
}


impl<State, Rng, ISTL, ILTL, ILTS> MonteCarlo<State> for HybridMonteCarlo<State, Rng, ISTL, ILTL, ILTS>
    where State: LatticeStateNew + Clone,
    ISTL: Integrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTL: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTS: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, LatticeHamiltonianSimulationStateSyncDefault<State>> + Clone,
    Rng: rand::Rng,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_random_e_state(state, self.get_rng());
        self.internal.get_next_element_default(state_internal, &mut self.rng).map(|el| el.get_state_owned())
    }
}


#[derive(Clone, Debug, PartialEq)]
struct HybridMonteCarloInternal<State, ISTL, ILTL, ILTS>
    where State: SimulationStateSynchrone,
    ISTL: Integrator<State, SimulationStateLeap<State>> + Clone,
    ILTL: Integrator<SimulationStateLeap<State>, SimulationStateLeap<State>> + Clone,
    ILTS: Integrator<SimulationStateLeap<State>, State> + Clone,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator_to_leap: ISTL,
    integrator_leap: ILTL,
    integrator_to_sync: ILTS,
    _phantom: PhantomData<State>,
}

impl<State, ISTL, ILTL, ILTS> HybridMonteCarloInternal<State, ISTL, ILTL, ILTS>
    where State: SimulationStateSynchrone,
    ISTL: Integrator<State, SimulationStateLeap<State>> + Clone,
    ILTL: Integrator<SimulationStateLeap<State>, SimulationStateLeap<State>> + Clone,
    ILTS: Integrator<SimulationStateLeap<State>, State> + Clone,
{
    pub fn new(
        delta_t: Real,
        number_of_steps: usize,
        integrator_to_leap: ISTL,
        integrator_leap: ILTL,
        integrator_to_sync: ILTS,
    ) -> Self {
        Self {
            delta_t,
            number_of_steps,
            integrator_to_leap,
            integrator_leap,
            integrator_to_sync,
            _phantom: PhantomData,
        }
    }
}

impl<State, ISTL, ILTL, ILTS> MonteCarloDefault<State> for HybridMonteCarloInternal<State, ISTL, ILTL, ILTS>
    where State: SimulationStateSynchrone,
    ISTL: Integrator<State, SimulationStateLeap<State>> + Clone,
    ILTL: Integrator<SimulationStateLeap<State>, SimulationStateLeap<State>> + Clone,
    ILTS: Integrator<SimulationStateLeap<State>, State> + Clone,
{
    
    fn get_potential_next_element(&mut self, state: &State, _rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        state.simulate_state_leapfrog_n(self.delta_t, self.number_of_steps, &self.integrator_to_leap, &self.integrator_leap, &self.integrator_to_sync)
    }
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (- old_state.get_hamiltonian_total() + new_state.get_hamiltonian_total()).exp().min(1_f64)
    }
    
}


pub struct MetropolisHastings<State>
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
    
    fn get_potential_next_element(&mut self, state: &State, rng: &mut impl rand::Rng) -> Result<State, SimulationError> {
        todo!()
    }
}
