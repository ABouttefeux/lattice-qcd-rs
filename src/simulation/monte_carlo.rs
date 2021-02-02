
use super::{
    super::{
        Real,
        integrator::Integrator,
    },
    state::{
        LatticeSimulationState,
        SimulationStateSynchrone,
        SimulationStateLeap,
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
    where State: LatticeSimulationState,
    RNG: rand::Rng,
{
    
    
    fn get_potential_next_element(&mut self, state: &State) -> Result<State, SimulationError>;
    
    fn get_rng(&mut self) -> &mut RNG;
    
    fn get_probability_of_replacement(old_state: &State, new_state : &State) -> Real {
        old_state.get_probability_of_next_element(new_state)
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

impl<State, RNG, ISTL, ILTL, ILTS> MonteCarlo<State, RNG> for HybridMonteCarlo<State, RNG, ISTL, ILTL, ILTS>
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
}
