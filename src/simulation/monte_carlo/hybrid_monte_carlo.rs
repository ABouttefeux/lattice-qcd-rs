//! Hybrid Monte Carlo methode
//!
//! # Example
//! ```
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use lattice_qcd_rs::integrator::SymplecticEulerRayon;
//! use lattice_qcd_rs::simulation::{
//!     HybridMonteCarloDiagnostic, LatticeState, LatticeStateDefault,
//! };
//! use rand::SeedableRng;
//!
//! let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
//! let mut hmc =
//!     HybridMonteCarloDiagnostic::new(0.000_000_1_f64, 10, SymplecticEulerRayon::new(), rng);
//! // Realistically you want more steps than 10
//!
//! let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 6_f64, 4)?;
//! for _ in 0..1 {
//!     state = state.monte_carlo_step(&mut hmc)?;
//!     println!(
//!         "probability of accept last step {}, has replaced {}",
//!         hmc.prob_replace_last(),
//!         hmc.has_replace_last()
//!     );
//!     // operation to track the progress or the evolution
//! }
//! // operation at the end of the simulation
//! #     Ok(())
//! # }
//! ```

use std::marker::PhantomData;

use rand_distr::Distribution;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        super::{error::MultiIntegrationError, integrator::SymplecticIntegrator, Real},
        state::{
            LatticeState, LatticeStateWithEFieldSyncDefault, SimulationStateLeap,
            SimulationStateSynchrone,
        },
    },
    MonteCarlo, MonteCarloDefault,
};

/// Hybrid Monte Carlo algorithm (HCM for short).
///
/// The idea of HCM is to generate a random set on conjugate momenta to the link matrices.
/// This conjugatewd momenta is also refed as the "Electric" field
/// or `e_field` with distribution N(0, 1 / beta). And to solve the equation of motion.
/// The new state is accepted with probability Exp( -H_old + H_new) where the Hamiltonian has an extra term Tr(E_i ^ 2).
/// The advantage is that the simulation can be done in a simpleptic way i.e. it conserved the Hamiltonian.
/// Which means that the methode has a high acceptance rate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HybridMonteCarlo<State, Rng, I, const D: usize>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    internal: HybridMonteCarloInternal<LatticeStateWithEFieldSyncDefault<State, D>, I, D>,
    rng: Rng,
}

impl<State, Rng, I, const D: usize> HybridMonteCarlo<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    getter!(
        /// Get a ref to the rng.
        pub rng() -> Rng
    );

    project!(
        /// Get the integrator.
        pub internal.integrator() -> &I
    );

    project_mut!(
        /// Get a mut ref to the integrator.
        pub internal.integrator_mut() -> &mut I
    );

    project!(
        /// Get `delta_t`.
        pub internal.delta_t() -> Real
    );

    project!(
        /// Get the number of steps.
        pub internal.number_of_steps() -> usize
    );

    /// gvies the following parameter for the HCM :
    /// - delta_t is the step size per intgration of the equation of motion
    /// - number_of_steps is the number of time
    /// - integrator is the methode to solve the equation of motion
    /// - rng, a random number generator
    pub fn new(delta_t: Real, number_of_steps: usize, integrator: I, rng: Rng) -> Self {
        Self {
            internal:
                HybridMonteCarloInternal::<LatticeStateWithEFieldSyncDefault<State, D>, I, D>::new(
                    delta_t,
                    number_of_steps,
                    integrator,
                ),
            rng,
        }
    }

    /// Get a mutlable reference to the rng.
    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// Get the last probably of acceptance of the random change.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
}

impl<State, Rng, I, const D: usize> AsRef<Rng> for HybridMonteCarlo<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    fn as_ref(&self) -> &Rng {
        self.rng()
    }
}

impl<State, Rng, I, const D: usize> AsMut<Rng> for HybridMonteCarlo<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    fn as_mut(&mut self) -> &mut Rng {
        self.rng_mut()
    }
}

impl<State, Rng, I, const D: usize> MonteCarlo<State, D> for HybridMonteCarlo<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    type Error = MultiIntegrationError<I::Error>;

    fn next_element(&mut self, state: State) -> Result<State, Self::Error> {
        let state_internal = LatticeStateWithEFieldSyncDefault::<State, D>::new_random_e_state(
            state,
            self.rng_mut(),
        );
        self.internal
            .next_element_default(state_internal, &mut self.rng)
            .map(|el| el.state_owned())
    }
}

/// internal structure for HybridMonteCarlo using [`LatticeStateWithEField`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
struct HybridMonteCarloInternal<State, I, const D: usize>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator: I,
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    _phantom: PhantomData<State>,
}

impl<State, I, const D: usize> HybridMonteCarloInternal<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    getter!(
        /// Get the integrator.
        pub,
        integrator,
        I
    );

    getter_copy!(
        /// Get `delta_t`.
        pub,
        delta_t,
        Real
    );

    getter_copy!(
        /// Get the number of steps.
        pub,
        number_of_steps,
        usize
    );

    /// see [HybridMonteCarlo::new]
    pub fn new(delta_t: Real, number_of_steps: usize, integrator: I) -> Self {
        Self {
            delta_t,
            number_of_steps,
            integrator,
            _phantom: PhantomData,
        }
    }

    pub fn integrator_mut(&mut self) -> &mut I {
        &mut self.integrator
    }
}

impl<State, I, const D: usize> AsRef<I> for HybridMonteCarloInternal<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    fn as_ref(&self) -> &I {
        self.integrator()
    }
}

impl<State, I, const D: usize> AsMut<I> for HybridMonteCarloInternal<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    fn as_mut(&mut self) -> &mut I {
        self.integrator_mut()
    }
}

impl<State, I, const D: usize> MonteCarloDefault<State, D> for HybridMonteCarloInternal<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    type Error = MultiIntegrationError<I::Error>;

    fn potential_next_element<Rng>(
        &mut self,
        state: &State,
        _rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
        state.simulate_symplectic_n_auto(&self.integrator, self.delta_t, self.number_of_steps)
    }

    fn probability_of_replacement(old_state: &State, new_state: &State) -> Real {
        (old_state.hamiltonian_total() - new_state.hamiltonian_total())
            .exp()
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
pub struct HybridMonteCarloDiagnostic<State, Rng, I, const D: usize>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    internal:
        HybridMonteCarloInternalDiagnostics<LatticeStateWithEFieldSyncDefault<State, D>, I, D>,
    rng: Rng,
}

impl<State, Rng, I, const D: usize> HybridMonteCarloDiagnostic<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    getter!(
        /// Get a ref to the rng.
        pub,
        rng,
        Rng
    );

    project!(
        /// Get the integrator.
        pub,
        integrator,
        internal,
        &I
    );

    project_mut!(
        /// Get a mut ref to the integrator.
        pub,
        integrator_mut,
        internal,
        &mut I
    );

    project!(
        /// Get `delta_t`.
        pub,
        delta_t,
        internal,
        Real
    );

    project!(
        /// Get the number of steps.
        pub,
        number_of_steps,
        internal,
        usize
    );

    /// gvies the following parameter for the HCM :
    /// - delta_t is the step size per intgration of the equation of motion
    /// - number_of_steps is the number of time
    /// - integrator is the methode to solve the equation of motion
    /// - rng, a random number generator
    pub fn new(delta_t: Real, number_of_steps: usize, integrator: I, rng: Rng) -> Self {
        Self {
            internal: HybridMonteCarloInternalDiagnostics::<
                LatticeStateWithEFieldSyncDefault<State, D>,
                I,
                D,
            >::new(delta_t, number_of_steps, integrator),
            rng,
        }
    }

    /// Get a mutlable reference to the rng.
    pub fn rng_mut(&mut self) -> &mut Rng {
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

impl<State, Rng, I, const D: usize> AsRef<Rng> for HybridMonteCarloDiagnostic<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    fn as_ref(&self) -> &Rng {
        self.rng()
    }
}

impl<State, Rng, I, const D: usize> AsMut<Rng> for HybridMonteCarloDiagnostic<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    fn as_mut(&mut self) -> &mut Rng {
        self.rng_mut()
    }
}

impl<State, Rng, I, const D: usize> MonteCarlo<State, D>
    for HybridMonteCarloDiagnostic<State, Rng, I, D>
where
    State: LatticeState<D> + Clone,
    LatticeStateWithEFieldSyncDefault<State, D>: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<
        LatticeStateWithEFieldSyncDefault<State, D>,
        SimulationStateLeap<LatticeStateWithEFieldSyncDefault<State, D>, D>,
        D,
    >,
    Rng: rand::Rng,
{
    type Error = MultiIntegrationError<I::Error>;

    fn next_element(&mut self, state: State) -> Result<State, Self::Error> {
        let state_internal = LatticeStateWithEFieldSyncDefault::<State, D>::new_random_e_state(
            state,
            self.rng_mut(),
        );
        self.internal
            .next_element_default(state_internal, &mut self.rng)
            .map(|el| el.state_owned())
    }
}

/// internal structure for HybridMonteCarlo using [`LatticeStateWithEField`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
struct HybridMonteCarloInternalDiagnostics<State, I, const D: usize>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    delta_t: Real,
    number_of_steps: usize,
    integrator: I,
    has_replace_last: bool,
    prob_replace_last: Real,
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    _phantom: PhantomData<State>,
}

impl<State, I, const D: usize> HybridMonteCarloInternalDiagnostics<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    getter!(
        /// Get the integrator.
        pub,
        integrator,
        I
    );

    getter_copy!(
        /// Get `delta_t`.
        pub,
        delta_t,
        Real
    );

    getter_copy!(
        /// Get the number of steps.
        pub,
        number_of_steps,
        usize
    );

    /// see [HybridMonteCarlo::new]
    pub fn new(delta_t: Real, number_of_steps: usize, integrator: I) -> Self {
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

    pub fn integrator_mut(&mut self) -> &mut I {
        &mut self.integrator
    }
}

impl<State, I, const D: usize> AsRef<I> for HybridMonteCarloInternalDiagnostics<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    fn as_ref(&self) -> &I {
        self.integrator()
    }
}

impl<State, I, const D: usize> AsMut<I> for HybridMonteCarloInternalDiagnostics<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    fn as_mut(&mut self) -> &mut I {
        self.integrator_mut()
    }
}

impl<State, I, const D: usize> MonteCarloDefault<State, D>
    for HybridMonteCarloInternalDiagnostics<State, I, D>
where
    State: SimulationStateSynchrone<D>,
    I: SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>,
{
    type Error = MultiIntegrationError<I::Error>;

    fn potential_next_element<Rng>(
        &mut self,
        state: &State,
        _rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
        state.simulate_symplectic_n_auto(&self.integrator, self.delta_t, self.number_of_steps)
    }

    fn probability_of_replacement(old_state: &State, new_state: &State) -> Real {
        (old_state.hamiltonian_total() - new_state.hamiltonian_total())
            .exp()
            .min(1_f64)
            .max(0_f64)
    }

    fn next_element_default<Rng>(
        &mut self,
        state: State,
        rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
        let potential_next = self.potential_next_element(&state, rng)?;
        let proba = Self::probability_of_replacement(&state, &potential_next)
            .min(1_f64)
            .max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            self.has_replace_last = true;
            Ok(potential_next)
        }
        else {
            self.has_replace_last = false;
            Ok(state)
        }
    }
}
