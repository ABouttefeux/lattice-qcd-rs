
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
pub struct HybridMonteCarlo<State, Rng, I>
    where State: LatticeStateNew + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    internal: HybridMonteCarloInternal<LatticeHamiltonianSimulationStateSyncDefault<State>, I>,
    rng: Rng,
}

impl<State, Rng, I> HybridMonteCarlo<State, Rng, I>
    where State: LatticeStateNew + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
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
    where State: LatticeStateNew + Clone,
    I: SymplecticIntegrator<LatticeHamiltonianSimulationStateSyncDefault<State>, SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<State>>>,
    Rng: rand::Rng,
{
    fn get_next_element(&mut self, state: State) -> Result<State, SimulationError> {
        let state_internal = LatticeHamiltonianSimulationStateSyncDefault::<State>::new_random_e_state(state, self.get_rng());
        self.internal.get_next_element_default(state_internal, &mut self.rng).map(|el| el.get_state_owned())
    }
}


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
