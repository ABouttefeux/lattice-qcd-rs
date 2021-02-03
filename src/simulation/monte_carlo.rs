
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
        LatticeHamiltonianSimulationStateNew,
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

pub trait MonteCarlo<State, RNG>
    where State: LatticeState,
    RNG: rand::Rng,
{
    fn get_potential_next_element(&mut self, state: &State) -> Result<State, SimulationError>;
    
    fn get_rng(&mut self) -> &mut RNG;
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (- old_state.get_hamiltonian_links() + new_state.get_hamiltonian_links()).exp().min(1_f64)
    }
    
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let potential_next = self.get_potential_next_element(&state)?;
        let r = DISTRIBUTION_0_1.sample(self.get_rng());
        if r < Self::get_probability_of_replacement(&state, &potential_next) {
            return Ok(potential_next);
        }
        else{
            return Ok(state);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HybridMonteCarlo<State, RNG, ISTL, ILTL, ILTS>
    where State: LatticeStateNew + Clone,
    ISTL: Integrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTL: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTS: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, LatticeHamiltonianSimulationStateSyncDefault<State>> + Clone,
    RNG: rand::Rng,
{
    internal: HybridMonteCarloInternal<LatticeHamiltonianSimulationStateSyncDefault<State>, RNG, ISTL, ILTL, ILTS>
}

impl<State, RNG, ISTL, ILTL, ILTS> HybridMonteCarlo<State, RNG, ISTL, ILTL, ILTS>
    where State: LatticeStateNew + Clone,
    ISTL: Integrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTL: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTS: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, LatticeHamiltonianSimulationStateSyncDefault<State>> + Clone,
    RNG: rand::Rng,
{
    pub fn new(
        delta_t: Real,
        number_of_steps: usize,
        integrator_to_leap: ISTL,
        integrator_leap: ILTL,
        integrator_to_sync: ILTS,
        rng: RNG,
    ) -> Self {
        Self {
            internal: HybridMonteCarloInternal::<LatticeHamiltonianSimulationStateSyncDefault<State>, RNG, ISTL, ILTL, ILTS>::new(delta_t, number_of_steps, integrator_to_leap, integrator_leap, integrator_to_sync, rng)
        }
    }
}

impl<State, RNG, ISTL, ILTL, ILTS> MonteCarlo<State, RNG> for HybridMonteCarlo<State, RNG, ISTL, ILTL, ILTS>
    where State: LatticeStateNew + Clone,
    ISTL: Integrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTL: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>> + Clone,
    ILTS: Integrator<SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>, LatticeHamiltonianSimulationStateSyncDefault<State>> + Clone,
    RNG: rand::Rng,
{
    fn get_rng(&mut self) -> &mut RNG {
        self.internal.get_rng()
    }
    
    fn get_potential_next_element(&mut self, _state: &State) -> Result<State, SimulationError> {
        unreachable!()
    }
    
    // the probablty of the next element is given by the internal Monte Carlo
    fn get_probability_of_replacement(_old_state: &State, _new_state : &State) -> Real {
        unreachable!()
    }
    
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_random_e_state(state, self.get_rng());
        self.internal.get_next_element(state_internal).map(|el|el.get_state_owned())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct HybridMonteCarloInternal<State, RNG, ISTL, ILTL, ILTS>
    where State: SimulationStateSynchrone,
    ISTL: Integrator<State, SimulationStateLeap<State>> + Clone,
    ILTL: Integrator<SimulationStateLeap<State>, SimulationStateLeap<State>> + Clone,
    ILTS: Integrator<SimulationStateLeap<State>, State> + Clone,
    RNG: rand::Rng,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator_to_leap: ISTL,
    integrator_leap: ILTL,
    integrator_to_sync: ILTS,
    rng: RNG,
    _phantom: PhantomData<State>,
}

impl<State, RNG, ISTL, ILTL, ILTS> HybridMonteCarloInternal<State, RNG, ISTL, ILTL, ILTS>
    where State: SimulationStateSynchrone,
    ISTL: Integrator<State, SimulationStateLeap<State>> + Clone,
    ILTL: Integrator<SimulationStateLeap<State>, SimulationStateLeap<State>> + Clone,
    ILTS: Integrator<SimulationStateLeap<State>, State> + Clone,
    RNG: rand::Rng,
{
    pub fn new(
        delta_t: Real,
        number_of_steps: usize,
        integrator_to_leap: ISTL,
        integrator_leap: ILTL,
        integrator_to_sync: ILTS,
        rng: RNG,
    ) -> Self {
        Self {
            delta_t,
            number_of_steps,
            integrator_to_leap,
            integrator_leap,
            integrator_to_sync,
            rng,
            _phantom: PhantomData,
        }
    }
}

impl<State, RNG, ISTL, ILTL, ILTS> MonteCarlo<State, RNG> for HybridMonteCarloInternal<State, RNG, ISTL, ILTL, ILTS>
    where State: SimulationStateSynchrone,
    ISTL: Integrator<State, SimulationStateLeap<State>> + Clone,
    ILTL: Integrator<SimulationStateLeap<State>, SimulationStateLeap<State>> + Clone,
    ILTS: Integrator<SimulationStateLeap<State>, State> + Clone,
    RNG: rand::Rng,
{
    fn get_rng(&mut self) -> &mut RNG {
        &mut self.rng
    }
    
    fn get_potential_next_element(&mut self, state: &State) -> Result<State, SimulationError> {
        state.simulate_state_leapfrog_n(self.delta_t, self.number_of_steps, &self.integrator_to_leap, &self.integrator_leap, &self.integrator_to_sync)
    }
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        (- old_state.get_hamiltonian_total() + new_state.get_hamiltonian_total()).exp().min(1_f64)
    }
}

pub struct MetropolisHastings<State, RNG>
    where State: LatticeState,
    RNG: rand::Rng,
{
    rng: RNG,
    _phantom: PhantomData<State>,
}

impl<State, RNG> MetropolisHastings<State, RNG>
    where State: LatticeState,
    RNG: rand::Rng,
{
    pub fn new(rng: RNG) -> Self {
        Self {rng, _phantom: PhantomData}
    }
}


impl<State, RNG> MonteCarlo<State, RNG> for MetropolisHastings<State, RNG>
    where State: LatticeState,
    RNG: rand::Rng,
{
    fn get_rng(&mut self) -> &mut RNG {
        &mut self.rng
    }
    
    fn get_potential_next_element(&mut self, state: &State) -> Result<State, SimulationError> {
        todo!()
    }
}
