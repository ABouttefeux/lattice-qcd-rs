//! Module containing the simulation State.
//!
//! The trait [`LatticeState`] is the most basic state with just the link matrices.
//! The default implementation is [`LatticeStateDefault`].
//! More over a more advance state with the color electrical field is given by
//! the trait [`LatticeStateWithEField`] with the default implementation
//! [`LatticeStateEFSyncDefault`]. Thw word "Sync" appears. It means that the simulation
//! for the equation of movement are done at the same time for both the link matrices and
//! the electric field. It is given by te trait [`SimulationStateSynchronous`]. It can also
//! be done using the leapfrog method with the trait [`SimulationStateLeapFrog`]
//! and the wrapper [`SimulationStateLeap`].
//!
//! See item documentation for more details.

use crossbeam::thread;
use na::{ComplexField, SVector};
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        error::{
            LatticeInitializationError, MultiIntegrationError, StateInitializationError,
            ThreadedStateInitializationError,
        },
        field::{EField, LinkMatrix, Su3Adjoint},
        integrator::SymplecticIntegrator,
        lattice::{
            Direction, DirectionList, LatticeCyclic, LatticeElementToIndex, LatticeLink,
            LatticeLinkCanonical, LatticePoint,
        },
        su3,
        thread::{ThreadAnyError, ThreadError},
        CMatrix3, Complex, Real, Vector8,
    },
    monte_carlo::MonteCarlo,
};

/// Default leap frog simulation state
pub type LeapFrogStateDefault<const D: usize> =
    SimulationStateLeap<LatticeStateEFSyncDefault<LatticeStateDefault<D>, D>, D>;

/// Trait to represent a pure gauge lattice state of dimension `D`.
///
/// It defines only one field: `link_matrix` of type [`LinkMatrix`].
///
/// # Example
/// They are many examples throughout the carte see by instance
/// [`super::monte_carlo::hybrid_monte_carlo`].
pub trait LatticeState<const D: usize> {
    /// Get the link matrices of this state.
    ///
    /// This is the field that stores the link matrices.
    /// # Example
    /// ```
    /// use lattice_qcd_rs::lattice::{DirectionEnum, LatticePoint};
    /// use lattice_qcd_rs::simulation::{LatticeState, LatticeStateDefault};
    ///
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let point = LatticePoint::new_zero();
    /// let state = LatticeStateDefault::<4>::new_cold(1_f64, 10_f64, 4)?;
    /// let _plaquette = state.link_matrix().pij(
    ///     &point,
    ///     &DirectionEnum::XPos.into(),
    ///     &DirectionEnum::YPos.into(),
    ///     state.lattice(),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn link_matrix(&self) -> &LinkMatrix;

    /// Replace the links matrices with the given input. It should panic if link matrix
    /// is not of the correct size.
    ///
    /// # Panic
    /// Panic if the length of link_matrix is different from
    /// `lattice.get_number_of_canonical_links_space()`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix);

    /// Get the lattice into which the state exists.
    fn lattice(&self) -> &LatticeCyclic<D>;

    /// Returns the beta parameter of the states.
    fn beta(&self) -> Real;

    /// C_A constant of the model, usually it is 3.
    const CA: Real;

    /// Returns the Hamiltonian of the links configuration.
    fn hamiltonian_links(&self) -> Real;

    /// Do one monte carlo step with the given method.
    ///
    /// # Errors
    /// The error form `MonteCarlo::get_next_element` is propagated.
    ///
    /// # Example
    /// see [`super::monte_carlo::hybrid_monte_carlo`].
    fn monte_carlo_step<M>(self, m: &mut M) -> Result<Self, M::Error>
    where
        Self: Sized,
        M: MonteCarlo<Self, D> + ?Sized,
    {
        m.next_element(self)
    }

    /// Take the average of the trace of all plaquettes.
    ///
    /// # Example
    /// see the crate documentation [`crate`].
    fn average_trace_plaquette(&self) -> Option<Complex> {
        self.link_matrix().average_trace_plaquette(self.lattice())
    }
}

/// Trait for a way to create a [`LatticeState`] from some parameters.
///
/// It is separated from the [`LatticeState`] because not all [`LatticeState`] can be create in this way.
/// By instance when there is also a field of conjugate momenta of the link matrices.
///
/// This is used by the Monte Carlo algorithms to create the new states.
pub trait LatticeStateNew<const D: usize>: LatticeState<D> + Sized {
    /// Error type
    type Error;

    /// Create a new simulation state.
    ///
    /// # Errors
    /// Give an error if the parameter are incorrect or the length of `link_matrix` does not correspond
    /// to `lattice`.
    ///
    /// # Example
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use lattice_qcd_rs::field::LinkMatrix;
    /// use lattice_qcd_rs::lattice::LatticeCyclic;
    /// use lattice_qcd_rs::simulation::{LatticeStateDefault, LatticeStateNew};
    ///
    /// let lattice = LatticeCyclic::new(1_f64, 4)?;
    /// let links = LinkMatrix::new_cold(&lattice);
    /// let state = LatticeStateDefault::<4>::new(lattice, 1_f64, links)?;
    /// # Ok(())
    /// # }
    /// ```
    fn new(
        lattice: LatticeCyclic<D>,
        beta: Real,
        link_matrix: LinkMatrix,
    ) -> Result<Self, Self::Error>;
}

/// Represent a lattice state where the conjugate momenta of the link matrices are included.
///
/// If you have a LatticeState and want the default way of adding the conjugate momenta look at
/// [`LatticeStateEFSyncDefault`].
///
/// If you want to solve the equation of motion using an [`SymplecticIntegrator`] also implement
/// [`SimulationStateSynchronous`] and the wrapper [`SimulationStateLeap`] can give you an [`SimulationStateLeapFrog`].
///
/// It is used for the [`super::monte_carlo::HybridMonteCarlo`] algorithm.
pub trait LatticeStateWithEField<const D: usize>: LatticeState<D> {
    /// Reset the e_field with radom value distributed as N(0, 1 / beta)
    /// [`rand_distr::StandardNormal`].
    ///
    /// # Errors
    /// Gives and error if N(0, 0.5/beta ) is not a valid distribution (for example beta = 0).
    ///
    /// Gives [`StateInitializationError::GaussProjectionError`] if the Gauss projection failed
    // TODO explain why !!!
    fn reset_e_field<Rng>(&mut self, rng: &mut Rng) -> Result<(), StateInitializationError>
    where
        Rng: rand::Rng + ?Sized,
    {
        let d = rand_distr::Normal::new(0_f64, 0.5_f64 / self.beta())?;
        let new_e_field = EField::new_determinist(self.lattice(), rng, &d);
        if !self.lattice().has_compatible_length_e_field(&new_e_field) {
            return Err(StateInitializationError::IncompatibleSize);
        }
        self.set_e_field(
            new_e_field
                .project_to_gauss(self.link_matrix(), self.lattice())
                .ok_or(StateInitializationError::GaussProjectionError)?,
        );
        Ok(())
    }

    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField<D>;

    /// Replace the electrical field with the given input. It should panic if the input is not of
    /// the correct size.
    ///
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_points()`
    fn set_e_field(&mut self, e_field: EField<D>);

    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize;

    /// Get the derivative \partial_t U(link), returns [`None`] if the link is outside of the lattice.
    ///
    /// It is used in order to apply the equation of motion.
    fn derivative_u(
        link: &LatticeLinkCanonical<D>,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3>;

    /// Get the derivative \partial_t E(point), returns [`None`] if the link is outside of the lattice.
    ///
    /// It is used in order to apply the equation of motion.
    fn derivative_e(
        point: &LatticePoint<D>,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<SVector<Su3Adjoint, D>>;

    /// Get the energy of the conjugate momenta configuration
    fn hamiltonian_efield(&self) -> Real;

    /// Get the total energy, by default [`LatticeStateWithEField::hamiltonian_efield`]
    /// + [`LatticeState::hamiltonian_links`]
    fn hamiltonian_total(&self) -> Real {
        self.hamiltonian_links() + self.hamiltonian_efield()
    }
}

/// Trait to create a simulation state.
///
/// It is used by the [`super::monte_carlo::HybridMonteCarlo`] algorithm to create new state.
pub trait LatticeStateWithEFieldNew<const D: usize>
where
    Self: LatticeStateWithEField<D> + Sized,
{
    /// Error type
    type Error: From<rand_distr::NormalError>;

    /// Create a new simulation state
    ///
    /// # Errors
    /// Give an error if the parameter are incorrect or the length of `link_matrix`
    /// and `e_field` does not correspond to `lattice`
    fn new(
        lattice: LatticeCyclic<D>,
        beta: Real,
        e_field: EField<D>,
        link_matrix: LinkMatrix,
        t: usize,
    ) -> Result<Self, Self::Error>;

    /// Create a new state with e_field randomly distributed as [`rand_distr::Normal`]^.
    ///
    /// # Errors
    /// Gives an error if N(0, 0.5/beta ) is not a valid distribution (for example beta = 0)
    /// or propagate the error from [`LatticeStateWithEFieldNew::new`]
    fn new_random_e<R>(
        lattice: LatticeCyclic<D>,
        beta: Real,
        link_matrix: LinkMatrix,
        rng: &mut R,
    ) -> Result<Self, Self::Error>
    where
        R: rand::Rng + ?Sized,
    {
        // TODO verify
        let d = rand_distr::Normal::new(0_f64, 0.5_f64 / beta)?;
        let e_field = EField::new_determinist(&lattice, rng, &d)
            .project_to_gauss(&link_matrix, &lattice)
            .expect("Projection to gauss failed");
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
}

/// [`LatticeStateWithEField`] who represent link matrices at the same time position as
/// its conjugate momenta
/// `e_field`.
///
/// If you have a LatticeState and want the default way of adding the conjugate momenta and doing
/// simulation look at
/// [`LatticeStateEFSyncDefault`].
///
/// I would advice of implementing this trait and not [`SimulationStateLeapFrog`], as there is
/// a wrapper ([`SimulationStateLeap`]) for [`SimulationStateLeapFrog`].
/// Also not implementing both trait gives you a compile time verification that you did not
/// considered a leap frog state as a sync one.
pub trait SimulationStateSynchronous<const D: usize>
where
    Self: LatticeStateWithEField<D> + Clone,
{
    /// does half a step for the conjugate momenta.
    ///
    /// # Errors
    /// Return an error if the integration could not be done.
    /// # Example
    /// see [`SimulationStateLeapFrog::simulate_leap`]
    fn simulate_to_leapfrog<I, State>(
        &self,
        integrator: &I,
        delta_t: Real,
    ) -> Result<State, I::Error>
    where
        State: SimulationStateLeapFrog<D>,
        I: SymplecticIntegrator<Self, State, D> + ?Sized,
    {
        integrator.integrate_sync_leap(self, delta_t)
    }

    /// Does `number_of_steps` with `delta_t` at each step using a leap_frog algorithm by fist
    /// doing half a step and then finishing by doing half step.
    ///
    /// # Errors
    /// Return an error if the integration could not be done
    /// or [`MultiIntegrationError::ZeroIntegration`] is the number of step is zero.
    // TODO example
    fn simulate_using_leapfrog_n<I, State>(
        &self,
        integrator: &I,
        delta_t: Real,
        number_of_steps: usize,
    ) -> Result<Self, MultiIntegrationError<I::Error>>
    where
        State: SimulationStateLeapFrog<D>,
        I: SymplecticIntegrator<Self, State, D> + ?Sized,
    {
        if number_of_steps == 0 {
            return Err(MultiIntegrationError::ZeroIntegration);
        }
        let mut state_leap = self
            .simulate_to_leapfrog(integrator, delta_t)
            .map_err(|error| MultiIntegrationError::IntegrationError(0, error))?;
        if number_of_steps > 1 {
            let result = state_leap.simulate_leap_n(integrator, delta_t, number_of_steps - 1);
            match result {
                Ok(state) => state_leap = state,
                Err(error) => {
                    match error {
                        MultiIntegrationError::IntegrationError(i, error) => {
                            return Err(MultiIntegrationError::IntegrationError(i + 1, error))
                        }
                        MultiIntegrationError::ZeroIntegration => {
                            // We cannot have 0 step integration as it is verified by the if
                            unreachable!();
                        }
                    }
                }
            }
        }
        let state_sync = state_leap
            .simulate_to_synchronous(integrator, delta_t)
            .map_err(|error| MultiIntegrationError::IntegrationError(number_of_steps, error))?;
        Ok(state_sync)
    }

    /// Does the same thing as [`SimulationStateSynchronous::simulate_using_leapfrog_n`]
    /// but use the default wrapper [`SimulationStateLeap`] for the leap frog state.
    ///
    /// # Errors
    /// Return an error if the integration could not be done.
    // TODO example
    fn simulate_using_leapfrog_n_auto<I>(
        &self,
        integrator: &I,
        delta_t: Real,
        number_of_steps: usize,
    ) -> Result<Self, MultiIntegrationError<I::Error>>
    where
        I: SymplecticIntegrator<Self, SimulationStateLeap<Self, D>, D> + ?Sized,
    {
        self.simulate_using_leapfrog_n(integrator, delta_t, number_of_steps)
    }

    /// Does a simulation step using the sync algorithm
    ///
    /// # Errors
    /// Return an error if the integration could not be done.
    // TODO example
    fn simulate_sync<I, T>(&self, integrator: &I, delta_t: Real) -> Result<Self, I::Error>
    where
        I: SymplecticIntegrator<Self, T, D> + ?Sized,
        T: SimulationStateLeapFrog<D>,
    {
        integrator.integrate_sync_sync(self, delta_t)
    }

    /// Does `numbers_of_times` of step of size `delta_t` using the sync algorithm
    ///
    /// # Errors
    /// Return an error if the integration could not be done
    /// or [`MultiIntegrationError::ZeroIntegration`] is the number of step is zero.
    // TODO example
    fn simulate_sync_n<I, T>(
        &self,
        integrator: &I,
        delta_t: Real,
        numbers_of_times: usize,
    ) -> Result<Self, MultiIntegrationError<I::Error>>
    where
        I: SymplecticIntegrator<Self, T, D> + ?Sized,
        T: SimulationStateLeapFrog<D>,
    {
        if numbers_of_times == 0 {
            return Err(MultiIntegrationError::ZeroIntegration);
        }
        let mut state = self
            .simulate_sync(integrator, delta_t)
            .map_err(|error| MultiIntegrationError::IntegrationError(0, error))?;
        for i in 1..numbers_of_times {
            state = state
                .simulate_sync(integrator, delta_t)
                .map_err(|error| MultiIntegrationError::IntegrationError(i, error))?;
        }
        Ok(state)
    }

    /// Integrate the state using the symplectic algorithm ( by going to leapfrog and back to sync)
    ///
    /// # Errors
    /// Return an error if the integration could not be done
    ///
    /// # Example
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use lattice_qcd_rs::integrator::{SymplecticEulerRayon, SymplecticIntegrator};
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField,
    ///     SimulationStateSynchronous,
    /// };
    /// use rand::SeedableRng;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
    /// let mut state = LatticeStateEFSyncDefault::new_random_e_state(
    ///     LatticeStateDefault::<3>::new_determinist(1_f64, 2_f64, 4, &mut rng)?,
    ///     &mut rng,
    /// );
    /// let h = state.hamiltonian_total();
    ///
    /// let integrator = SymplecticEulerRayon::default();
    /// for _ in 0..1 {
    ///     // Realistically you would want more steps
    ///     state = state.simulate_symplectic(&integrator, 0.000_001_f64)?;
    /// }
    /// let h2 = state.hamiltonian_total();
    ///
    /// println!("The error on the Hamiltonian is {}", h - h2);
    /// #     Ok(())
    /// # }
    /// ```
    fn simulate_symplectic<I, T>(&self, integrator: &I, delta_t: Real) -> Result<Self, I::Error>
    where
        I: SymplecticIntegrator<Self, T, D> + ?Sized,
        T: SimulationStateLeapFrog<D>,
    {
        integrator.integrate_symplectic(self, delta_t)
    }

    /// Does `numbers_of_times` of step of size `delta_t` using the symplectic algorithm
    ///
    /// # Errors
    /// Return an error if the integration could not be done
    /// or [`MultiIntegrationError::ZeroIntegration`] is the number of step is zero.
    // TODO example
    fn simulate_symplectic_n<I, T>(
        &self,
        integrator: &I,
        delta_t: Real,
        numbers_of_times: usize,
    ) -> Result<Self, MultiIntegrationError<I::Error>>
    where
        I: SymplecticIntegrator<Self, T, D> + ?Sized,
        T: SimulationStateLeapFrog<D>,
    {
        if numbers_of_times == 0 {
            return Err(MultiIntegrationError::ZeroIntegration);
        }
        let mut state = self
            .simulate_symplectic(integrator, delta_t)
            .map_err(|error| MultiIntegrationError::IntegrationError(0, error))?;
        for i in 1..numbers_of_times {
            state = state
                .simulate_symplectic(integrator, delta_t)
                .map_err(|error| MultiIntegrationError::IntegrationError(i, error))?;
        }
        Ok(state)
    }

    /// Does the same thing as [`SimulationStateSynchronous::simulate_symplectic_n`]
    /// but use the default wrapper [`SimulationStateLeap`] for the leap frog state.
    ///
    /// # Errors
    /// Return an error if the integration could not be done.
    // TODO example
    fn simulate_symplectic_n_auto<I>(
        &self,
        integrator: &I,
        delta_t: Real,
        number_of_steps: usize,
    ) -> Result<Self, MultiIntegrationError<I::Error>>
    where
        I: SymplecticIntegrator<Self, SimulationStateLeap<Self, D>, D> + ?Sized,
    {
        self.simulate_symplectic_n(integrator, delta_t, number_of_steps)
    }
}

/// [`LatticeStateWithEField`] who represent link matrices at time T and its conjugate
/// momenta at time T + 1/2.
///
/// If you have a [`SimulationStateSynchronous`] look at the wrapper [`SimulationStateLeap`].
pub trait SimulationStateLeapFrog<const D: usize>
where
    Self: LatticeStateWithEField<D>,
{
    /// Simulate the state to synchronous by finishing the half step.
    ///
    /// # Errors
    /// Return an error if the integration could not be done.
    ///
    /// # Example
    /// see [`SimulationStateLeapFrog::simulate_leap`]
    fn simulate_to_synchronous<I, State>(
        &self,
        integrator: &I,
        delta_t: Real,
    ) -> Result<State, I::Error>
    where
        Self: Sized,
        State: SimulationStateSynchronous<D> + ?Sized,
        I: SymplecticIntegrator<State, Self, D> + ?Sized,
    {
        integrator.integrate_leap_sync(self, delta_t)
    }

    /// Does one simulation step using the leap frog algorithm.
    ///
    /// # Errors
    /// Return an error if the integration could not be done.
    ///
    /// # Example
    ///  ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use lattice_qcd_rs::integrator::{SymplecticEulerRayon, SymplecticIntegrator};
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField, SimulationStateSynchronous, SimulationStateLeapFrog,
    /// };
    /// use rand::SeedableRng;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
    /// let state = LatticeStateEFSyncDefault::new_random_e_state(
    ///     LatticeStateDefault::<3>::new_determinist(1_f64, 2_f64, 4, &mut rng)?,
    ///     &mut rng,
    /// );
    /// let h = state.hamiltonian_total();
    /// let integrator = SymplecticEulerRayon::default();
    /// let mut leap_frog = state.simulate_to_leapfrog(&integrator,0.000_001_f64)?;
    /// drop(state);
    /// for _ in 0..2 {
    ///     // Realistically you would want more steps
    ///     leap_frog = leap_frog.simulate_leap(&integrator, 0.000_001_f64)?;
    /// }
    /// let state = leap_frog.simulate_to_synchronous(&integrator, 0.000_001_f64)?;
    /// let h2 = state.hamiltonian_total();
    ///
    /// println!("The error on the Hamiltonian is {}", h - h2);
    /// #     Ok(())
    /// # }
    /// ```
    fn simulate_leap<I, T>(&self, integrator: &I, delta_t: Real) -> Result<Self, I::Error>
    where
        Self: Sized,
        I: SymplecticIntegrator<T, Self, D> + ?Sized,
        T: SimulationStateSynchronous<D> + ?Sized,
    {
        integrator.integrate_leap_leap(self, delta_t)
    }

    /// does `numbers_of_times` simulation set of size `delta_t` using the leap frog algorithm.
    ///
    /// # Errors
    /// Return an error if the integration could not be done
    /// or [`MultiIntegrationError::ZeroIntegration`] is the number of step is zero.
    ///
    /// # Example
    /// /// # Example
    ///  ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use lattice_qcd_rs::integrator::{SymplecticEulerRayon, SymplecticIntegrator};
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField, SimulationStateSynchronous, SimulationStateLeapFrog,
    /// };
    /// use rand::SeedableRng;
    ///
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
    /// let state = LatticeStateEFSyncDefault::new_random_e_state(
    ///     LatticeStateDefault::<3>::new_determinist(1_f64, 2_f64, 4, &mut rng)?,
    ///     &mut rng,
    /// );
    /// let h = state.hamiltonian_total();
    /// let integrator = SymplecticEulerRayon::default();
    /// let mut leap_frog = state.simulate_to_leapfrog(&integrator,0.000_001_f64)?;
    /// drop(state);
    ///
    /// // Realistically you would want more steps
    /// leap_frog = leap_frog.simulate_leap_n(&integrator, 0.000_001_f64, 10)?;
    ///
    /// let state = leap_frog.simulate_to_synchronous(&integrator, 0.000_001_f64)?;
    /// let h2 = state.hamiltonian_total();
    ///
    /// println!("The error on the Hamiltonian is {}", h - h2);
    /// #     Ok(())
    /// # }
    /// ```
    fn simulate_leap_n<I, T>(
        &self,
        integrator: &I,
        delta_t: Real,
        numbers_of_times: usize,
    ) -> Result<Self, MultiIntegrationError<I::Error>>
    where
        Self: Sized,
        I: SymplecticIntegrator<T, Self, D> + ?Sized,
        T: SimulationStateSynchronous<D> + ?Sized,
    {
        if numbers_of_times == 0 {
            return Err(MultiIntegrationError::ZeroIntegration);
        }
        let mut state = self
            .simulate_leap(integrator, delta_t)
            .map_err(|error| MultiIntegrationError::IntegrationError(0, error))?;
        for i in 1..(numbers_of_times) {
            state = state
                .simulate_leap(integrator, delta_t)
                .map_err(|error| MultiIntegrationError::IntegrationError(i, error))?;
        }
        Ok(state)
    }
}

/// Represent a simulation state at a set time.
///
/// It has the default pure gauge hamiltonian
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeStateDefault<const D: usize> {
    lattice: LatticeCyclic<D>,
    beta: Real,
    link_matrix: LinkMatrix,
}

impl<const D: usize> LatticeStateDefault<D> {
    /// Create a cold configuration. i.e. all the links are set to the unit matrix.
    ///
    /// With the lattice of size `size` and dimension `number_of_points` ( see [`LatticeCyclic::new`] )
    /// and beta parameter `beta`.
    ///
    /// # Errors
    /// Returns [`StateInitializationError::LatticeInitializationError`] if the parameter is invalid
    /// for [`LatticeCyclic`].
    /// Or propagate the error form [`Self::new`].
    pub fn new_cold(
        size: Real,
        beta: Real,
        number_of_points: usize,
    ) -> Result<Self, StateInitializationError> {
        let lattice = LatticeCyclic::new(size, number_of_points)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        Self::new(lattice, beta, link_matrix)
    }

    /// Create a "hot" configuration, i.e. the link matrices are chosen randomly.
    ///
    /// With the lattice of size `size` and dimension `number_of_points` ( see [`LatticeCyclic::new`] )
    /// and beta parameter `beta`.
    ///
    /// The creation is determinists meaning that it is reproducible:
    ///
    /// # Errors
    /// Returns [`StateInitializationError::LatticeInitializationError`] if the parameter is invalid for [`LatticeCyclic`].
    /// Or propagate the error form [`Self::new`].
    ///
    /// # Example
    /// This example demonstrate how to reproduce the same configuration
    /// ```
    /// # use lattice_qcd_rs::{simulation::LatticeStateDefault, lattice::LatticeCyclic, dim};
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// assert_eq!(
    ///     LatticeStateDefault::<4>::new_determinist(1_f64, 1_f64, 4, &mut rng_1).unwrap(),
    ///     LatticeStateDefault::<4>::new_determinist(1_f64, 1_f64, 4, &mut rng_2).unwrap()
    /// );
    /// ```
    pub fn new_determinist(
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<Self, StateInitializationError> {
        let lattice = LatticeCyclic::new(size, number_of_points)?;
        let link_matrix = LinkMatrix::new_determinist(&lattice, rng);
        Self::new(lattice, beta, link_matrix)
    }

    /// Correct the numerical drift, reprojecting all the link matrices to SU(3).
    /// see [`LinkMatrix::normalize`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::error::ImplementationError;
    /// use lattice_qcd_rs::prelude::*;
    /// use rand::SeedableRng;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
    ///
    /// let size = 1_f64;
    /// let number_of_pts = 3;
    /// let beta = 1_f64;
    ///
    /// let mut simulation =
    ///     LatticeStateDefault::<4>::new_determinist(size, beta, number_of_pts, &mut rng)?;
    ///
    /// let spread_parameter = 0.1_f64;
    /// let mut mc = MetropolisHastingsSweep::new(1, spread_parameter, rng)
    ///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    ///
    /// for _ in 0..2 {
    ///     for _ in 0..10 {
    ///         simulation = simulation.monte_carlo_step(&mut mc)?;
    ///     }
    ///     // the more we advance te more the link matrices
    ///     // will deviate form SU(3), so we reproject to SU(3)
    ///     // every 10 steps.
    ///     simulation.normalize_link_matrices();
    /// }
    /// #
    /// # Ok(())
    /// # }
    /// ```
    pub fn normalize_link_matrices(&mut self) {
        self.link_matrix.normalize();
    }

    /// Get a mutable reference to the link matrix at `link`
    pub fn link_mut(&mut self, link: &LatticeLinkCanonical<D>) -> Option<&mut CMatrix3> {
        let index = link.to_index(&self.lattice);
        if index < self.link_matrix.len() {
            Some(&mut self.link_matrix[index])
        }
        else {
            None
        }
    }

    /// Absorbs self anf return the link_matrix as owned
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn link_matrix_owned(self) -> LinkMatrix {
        self.link_matrix
    }
}

impl<const D: usize> LatticeStateNew<D> for LatticeStateDefault<D> {
    type Error = StateInitializationError;

    fn new(
        lattice: LatticeCyclic<D>,
        beta: Real,
        link_matrix: LinkMatrix,
    ) -> Result<Self, Self::Error> {
        if !lattice.has_compatible_length_links(&link_matrix) {
            return Err(StateInitializationError::IncompatibleSize);
        }
        Ok(Self {
            lattice,
            beta,
            link_matrix,
        })
    }
}

impl<const D: usize> LatticeState<D> for LatticeStateDefault<D> {
    const CA: Real = 3_f64;

    getter!(
        /// The link matrices of this state.
        link_matrix,
        LinkMatrix
    );

    getter!(lattice, LatticeCyclic<D>);

    getter_copy!(beta, Real);

    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_canonical_links_space()`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        if self.lattice.number_of_canonical_links_space() != link_matrix.len() {
            panic!("Link matrices are not of the correct size");
        }
        self.link_matrix = link_matrix;
    }

    /// Get the default pure gauge Hamiltonian.
    ///
    /// # Panic
    /// Panic if plaquettes cannot be found
    fn hamiltonian_links(&self) -> Real {
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice()
            .get_points()
            .par_bridge()
            .map(|el| {
                Direction::positive_directions()
                    .iter()
                    .map(|dir_i| {
                        Direction::positive_directions()
                            .iter()
                            .filter(|dir_j| dir_i.index() < dir_j.index())
                            .map(|dir_j| {
                                1_f64
                                    - self
                                        .link_matrix()
                                        .pij(&el, dir_i, dir_j, self.lattice())
                                        .expect("Plaquette not found")
                                        .trace()
                                        .real()
                                        / Self::CA
                            })
                            .sum::<Real>()
                    })
                    .sum::<Real>()
            })
            .sum::<Real>()
            * self.beta()
    }
}

/// wrapper for a simulation state using leap frog ([`SimulationStateLeap`]) using a synchronous type
/// ([`SimulationStateSynchronous`]).
#[derive(Debug, PartialEq, Clone, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SimulationStateLeap<State, const D: usize>
where
    State: SimulationStateSynchronous<D> + ?Sized,
{
    state: State,
}

impl<State, const D: usize> SimulationStateLeap<State, D>
where
    State: SimulationStateSynchronous<D> + LatticeStateWithEField<D> + ?Sized,
{
    getter!(
        /// get a reference to the state
        pub const,
        state,
        State
    );

    /// Create a new SimulationStateLeap directly from a state without applying any modification.
    ///
    /// In most cases wou will prefer to build it using [`LatticeStateNew`] or [`Self::from_synchronous`].
    pub const fn new_from_state(state: State) -> Self {
        Self { state }
    }

    /// get a mutable reference to the state
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Create a leap state from a sync one by integrating by half a step the e_field.
    ///
    /// # Errors
    /// Returns an error if the integration failed.
    pub fn from_synchronous<I>(s: &State, integrator: &I, delta_t: Real) -> Result<Self, I::Error>
    where
        I: SymplecticIntegrator<State, Self, D> + ?Sized,
    {
        s.simulate_to_leapfrog(integrator, delta_t)
    }

    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn gauss(&self, point: &LatticePoint<D>) -> Option<CMatrix3> {
        self.e_field()
            .gauss(self.link_matrix(), point, self.lattice())
    }
}

impl<State, const D: usize> Default for SimulationStateLeap<State, D>
where
    State: SimulationStateSynchronous<D> + Default + ?Sized,
{
    fn default() -> Self {
        Self::new_from_state(State::default())
    }
}

impl<State, const D: usize> std::fmt::Display for SimulationStateLeap<State, D>
where
    State: SimulationStateSynchronous<D> + std::fmt::Display + ?Sized,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "leapfrog {}", self.state())
    }
}

impl<State: SimulationStateSynchronous<D> + LatticeStateWithEField<D>, const D: usize> AsRef<State>
    for SimulationStateLeap<State, D>
{
    fn as_ref(&self) -> &State {
        self.state()
    }
}

impl<State: SimulationStateSynchronous<D> + LatticeStateWithEField<D>, const D: usize> AsMut<State>
    for SimulationStateLeap<State, D>
{
    fn as_mut(&mut self) -> &mut State {
        self.state_mut()
    }
}

/// This state is a leap frog state
impl<State, const D: usize> SimulationStateLeapFrog<D> for SimulationStateLeap<State, D> where
    State: SimulationStateSynchronous<D> + LatticeStateWithEField<D> + ?Sized
{
}

/// We just transmit the function of `State`, there is nothing new.
impl<State, const D: usize> LatticeState<D> for SimulationStateLeap<State, D>
where
    State: LatticeStateWithEField<D> + SimulationStateSynchronous<D> + ?Sized,
{
    const CA: Real = State::CA;

    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        self.state().link_matrix()
    }

    /// # Panic
    /// panic under the same condition as `State::set_link_matrix`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        self.state.set_link_matrix(link_matrix);
    }

    fn lattice(&self) -> &LatticeCyclic<D> {
        self.state().lattice()
    }

    fn beta(&self) -> Real {
        self.state().beta()
    }

    fn hamiltonian_links(&self) -> Real {
        self.state().hamiltonian_links()
    }
}

impl<State, const D: usize> LatticeStateWithEFieldNew<D> for SimulationStateLeap<State, D>
where
    State: LatticeStateWithEField<D> + SimulationStateSynchronous<D> + LatticeStateWithEFieldNew<D>,
{
    type Error = State::Error;

    fn new(
        lattice: LatticeCyclic<D>,
        beta: Real,
        e_field: EField<D>,
        link_matrix: LinkMatrix,
        t: usize,
    ) -> Result<Self, Self::Error> {
        let state = State::new(lattice, beta, e_field, link_matrix, t)?;
        Ok(Self { state })
    }
}

/// We just transmit the function of `State`, there is nothing new.
impl<State, const D: usize> LatticeStateWithEField<D> for SimulationStateLeap<State, D>
where
    State: LatticeStateWithEField<D> + SimulationStateSynchronous<D> + ?Sized,
{
    project!(hamiltonian_efield, state, Real);

    project!(
        /// The "Electrical" field of this state.
        e_field,
        state,
        &EField<D>
    );

    project_mut!(
        /// # Panic
        /// panic under the same condition as `State::set_e_field`
        set_e_field,
        state,
        (),
        e_field: EField<D>
    );

    project!(
        /// return the time state, i.e. the number of time the simulation ran.
        t,
        state,
        usize
    );

    fn derivative_u(
        link: &LatticeLinkCanonical<D>,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3> {
        State::derivative_u(link, link_matrix, e_field, lattice)
    }

    fn derivative_e(
        point: &LatticePoint<D>,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<SVector<Su3Adjoint, D>> {
        State::derivative_e(point, link_matrix, e_field, lattice)
    }
}

/// wrapper to implement [`LatticeStateWithEField`] from a [`LatticeState`] using
/// the default implementation of conjugate momenta.
///
/// It also implement [`SimulationStateSynchronous`].
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeStateEFSyncDefault<State, const D: usize>
where
    State: LatticeState<D> + ?Sized,
{
    #[cfg_attr(
        feature = "serde-serialize",
        serde(bound(
            serialize = "SVector<Su3Adjoint, D>: Serialize",
            deserialize = "SVector<Su3Adjoint, D>: Deserialize<'de>"
        ))
    )]
    e_field: EField<D>,
    t: usize,
    lattice_state: State, // the DST must be at the end
}

impl<State, const D: usize> LatticeStateEFSyncDefault<State, D>
where
    State: LatticeState<D> + ?Sized,
{
    /// Absorbs self and return the state as owned.
    /// It essentially deconstruct the structure.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn state_owned(self) -> State
    where
        State: Sized,
    {
        self.lattice_state
    }

    /// Get a reference to the state.
    pub const fn lattice_state(&self) -> &State {
        &self.lattice_state
    }

    /// Get a mutable reference to the state.
    pub fn lattice_state_mut(&mut self) -> &mut State {
        &mut self.lattice_state
    }

    /// Take a state and generate a new random one and try projecting it to the Gauss law.
    ///
    /// # Panic
    /// Panics if N(0, 0.5/beta ) is not a valid distribution (for example beta = 0).
    /// Panics if the field could not be projected to the Gauss law.
    pub fn new_random_e_state(lattice_state: State, rng: &mut impl rand::Rng) -> Self
    where
        State: Sized,
    {
        let d = rand_distr::Normal::new(0_f64, 0.5_f64 / lattice_state.beta())
            .expect("Distribution not valid, check Beta.");
        let e_field = EField::new_determinist(lattice_state.lattice(), rng, &d)
            .project_to_gauss(lattice_state.link_matrix(), lattice_state.lattice())
            .unwrap();
        // TODO error management
        Self {
            lattice_state,
            e_field,
            t: 0,
        }
    }

    /// Create a new Self from a state and a cold configuration of the e field (i.e. set to 0)
    pub fn new_e_cold(lattice_state: State) -> Self
    where
        State: Sized,
    {
        let e_field = EField::new_cold(lattice_state.lattice());
        Self {
            lattice_state,
            e_field,
            t: 0,
        }
    }

    /// Get a mutable reference to the efield
    pub fn e_field_mut(&mut self) -> &mut EField<D> {
        &mut self.e_field
    }
}

impl<State, const D: usize> LatticeStateEFSyncDefault<State, D>
where
    Self: LatticeStateWithEField<D>,
    State: LatticeState<D> + ?Sized,
{
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn gauss(&self, point: &LatticePoint<D>) -> Option<CMatrix3> {
        self.e_field
            .gauss(self.link_matrix(), point, self.lattice())
    }
}

impl<State, const D: usize> LatticeStateEFSyncDefault<State, D>
where
    Self: LatticeStateWithEFieldNew<D>,
    <Self as LatticeStateWithEFieldNew<D>>::Error: From<LatticeInitializationError>,
    State: LatticeState<D>,
{
    /// Generate a hot (i.e. random) initial state.
    ///
    /// Single threaded generation with a given random number generator.
    /// `size` is the size parameter of the lattice and `number_of_points` is the number of points
    /// in each spatial dimension of the lattice. See [`LatticeCyclic::new`] for more info.
    ///
    /// useful to reproduce a set of data but slower than
    /// [`LatticeStateEFSyncDefault::new_random_threaded`].
    ///
    /// # Errors
    /// Return [`StateInitializationError::LatticeInitializationError`] if the parameter is invalid
    /// for [`LatticeCyclic`].
    /// Or propagates the error form [`Self::new`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{simulation::{LatticeStateEFSyncDefault, LatticeStateDefault}, lattice::LatticeCyclic};
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// assert_eq!(
    ///     LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(1_f64, 1_f64, 4, &mut rng_1, &distribution).unwrap(),
    ///     LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(1_f64, 1_f64, 4, &mut rng_2, &distribution).unwrap()
    /// );
    /// ```
    pub fn new_determinist<R>(
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut R,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Result<Self, <Self as LatticeStateWithEFieldNew<D>>::Error>
    where
        R: rand::Rng + ?Sized,
    {
        let lattice = LatticeCyclic::new(size, number_of_points)?;
        let e_field = EField::new_determinist(&lattice, rng, d);
        let link_matrix = LinkMatrix::new_determinist(&lattice, rng);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }

    /// Generate a configuration with cold e_field and hot link matrices
    ///
    /// # Errors
    /// Return [`StateInitializationError::LatticeInitializationError`] if the parameter is invalid
    /// for [`LatticeCyclic`].
    /// Or propagates the error form [`Self::new`].
    pub fn new_determinist_cold_e_hot_link<R>(
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut R,
    ) -> Result<Self, <Self as LatticeStateWithEFieldNew<D>>::Error>
    where
        R: rand::Rng + ?Sized,
    {
        let lattice = LatticeCyclic::new(size, number_of_points)?;
        let e_field = EField::new_cold(&lattice);
        let link_matrix = LinkMatrix::new_determinist(&lattice, rng);

        Self::new(lattice, beta, e_field, link_matrix, 0)
    }

    /// Generate a new cold state.
    ///
    /// It meas that the link matrices are set to the identity and electrical field are set to 0.
    ///
    /// # Errors
    /// Return [`StateInitializationError::LatticeInitializationError`] if the parameter is invalid
    /// for [`LatticeCyclic`].
    /// Or propagates the error form [`Self::new`].
    pub fn new_cold(
        size: Real,
        beta: Real,
        number_of_points: usize,
    ) -> Result<Self, <Self as LatticeStateWithEFieldNew<D>>::Error> {
        let lattice = LatticeCyclic::new(size, number_of_points)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        let e_field = EField::new_cold(&lattice);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
}

impl<State, const D: usize> LatticeStateEFSyncDefault<State, D>
where
    Self: LatticeStateWithEFieldNew<D, Error = StateInitializationError>,
    State: LatticeState<D>,
{
    /// Generate a hot (i.e. random) initial state.
    ///
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reproduce easily, In that case use
    /// [`LatticeStateEFSyncDefault::new_determinist`].
    ///
    /// # Errors
    /// Return [`StateInitializationError::LatticeInitializationError`] if the parameter is invalid
    /// for [`LatticeCyclic`].
    /// Return [`ThreadError::ThreadNumberIncorrect`] if `number_of_points = 0`.
    /// Returns an error if a thread panicked. Finally, propagates the error form [`Self::new`].
    pub fn new_random_threaded<Distribution>(
        size: Real,
        beta: Real,
        number_of_points: usize,
        d: &Distribution,
        number_of_thread: usize,
    ) -> Result<Self, ThreadedStateInitializationError>
    where
        Distribution: rand_distr::Distribution<Real> + Sync,
    {
        if number_of_thread == 0 {
            return Err(ThreadedStateInitializationError::ThreadingError(
                ThreadError::ThreadNumberIncorrect,
            ));
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return Self::new_determinist(size, beta, number_of_points, &mut rng, d)
                .map_err(|err| err.into());
        }
        let lattice = LatticeCyclic::new(size, number_of_points).map_err(|err| {
            ThreadedStateInitializationError::StateInitializationError(err.into())
        })?;
        thread::scope(|s| {
            let lattice_clone = lattice.clone();
            let handel = s.spawn(move |_| EField::new_random(&lattice_clone, d));
            let link_matrix = LinkMatrix::new_random_threaded(&lattice, number_of_thread - 1)?;
            let e_field = handel.join().map_err(|err| {
                ThreadedStateInitializationError::ThreadingError(
                    ThreadAnyError::Panic(vec![err]).into(),
                )
            })?;
            // TODO not very clean: improve
            Self::new(lattice, beta, e_field, link_matrix, 0)
                .map_err(ThreadedStateInitializationError::StateInitializationError)
        })
        .map_err(|err| {
            ThreadedStateInitializationError::ThreadingError(
                ThreadAnyError::Panic(vec![err]).into(),
            )
        })?
    }
}

/// This is an sync State
impl<State, const D: usize> SimulationStateSynchronous<D> for LatticeStateEFSyncDefault<State, D>
where
    State: LatticeState<D> + Clone + ?Sized,
    Self: LatticeStateWithEField<D>,
{
}

impl<State, const D: usize> LatticeState<D> for LatticeStateEFSyncDefault<State, D>
where
    State: LatticeState<D> + ?Sized,
{
    const CA: Real = State::CA;

    fn link_matrix(&self) -> &LinkMatrix {
        self.lattice_state.link_matrix()
    }

    /// # Panic
    /// panic under the same condition as `State::set_link_matrix`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        self.lattice_state.set_link_matrix(link_matrix);
    }

    fn lattice(&self) -> &LatticeCyclic<D> {
        self.lattice_state.lattice()
    }

    fn beta(&self) -> Real {
        self.lattice_state.beta()
    }

    fn hamiltonian_links(&self) -> Real {
        self.lattice_state.hamiltonian_links()
    }
}

impl<State, const D: usize> LatticeStateWithEFieldNew<D> for LatticeStateEFSyncDefault<State, D>
where
    State: LatticeState<D> + LatticeStateNew<D>,
    Self: LatticeStateWithEField<D>,
    StateInitializationError: Into<State::Error>,
    State::Error: From<rand_distr::NormalError>,
{
    type Error = State::Error;

    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    fn new(
        lattice: LatticeCyclic<D>,
        beta: Real,
        e_field: EField<D>,
        link_matrix: LinkMatrix,
        t: usize,
    ) -> Result<Self, Self::Error> {
        if !lattice.has_compatible_length_e_field(&e_field) {
            return Err(StateInitializationError::IncompatibleSize.into());
        }
        let lattice_state_r = State::new(lattice, beta, link_matrix);
        match lattice_state_r {
            Ok(lattice_state) => Ok(Self {
                e_field,
                t,
                lattice_state,
            }),
            Err(err) => Err(err),
        }
    }
}

impl<const D: usize> LatticeStateWithEField<D>
    for LatticeStateEFSyncDefault<LatticeStateDefault<D>, D>
where
    Direction<D>: DirectionList,
{
    /// By default \sum_x Tr(E_i E_i)
    fn hamiltonian_efield(&self) -> Real {
        self.lattice()
            .get_points()
            .par_bridge()
            .map(|el| {
                Direction::positive_directions()
                    .iter()
                    .map(|dir_i| {
                        let e_i = self.e_field().e_field(&el, dir_i, self.lattice()).unwrap();
                        e_i.trace_squared()
                    })
                    .sum::<Real>()
            })
            .sum::<Real>()
            * self.beta()
    }

    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField<D> {
        &self.e_field
    }

    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_points()`
    fn set_e_field(&mut self, e_field: EField<D>) {
        if self.lattice().number_of_points() != e_field.len() {
            panic!("e_field is not of the correct size");
        }
        self.e_field = e_field;
    }

    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.t
    }

    /// Get the derive of U_i(x).
    fn derivative_u(
        link: &LatticeLinkCanonical<D>,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, (2_f64 * Self::CA).sqrt());
        let u_i = link_matrix.matrix(&LatticeLink::from(*link), lattice)?;
        let e_i = e_field.e_field(link.pos(), link.dir(), lattice)?;
        Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / lattice.size()))
    }

    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    fn derivative_e(
        point: &LatticePoint<D>,
        link_matrix: &LinkMatrix,
        _e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<SVector<Su3Adjoint, D>> {
        let c = -(2_f64 / Self::CA).sqrt();
        let dir_pos = Direction::<D>::positive_directions();
        let iterator = dir_pos.iter().map(|dir| {
            let u_i = link_matrix.matrix(&LatticeLink::new(*point, *dir), lattice)?;
            let sum_s: CMatrix3 = Direction::<D>::directions()
                .iter()
                .filter(|dir_2| dir_2.to_positive() != *dir)
                .map(|dir_2| {
                    link_matrix
                        .sij(point, dir, dir_2, lattice)
                        .map(|el| el.adjoint())
                })
                .sum::<Option<CMatrix3>>()?;
            Some(Su3Adjoint::new(Vector8::<Real>::from_fn(|index, _| {
                c * (su3::GENERATORS[index] * u_i * sum_s).trace().imaginary() / lattice.size()
            })))
        });
        let mut return_vector = SVector::<_, D>::from_element(Su3Adjoint::default());
        for (index, element) in iterator.enumerate() {
            return_vector[index] = element?;
        }
        Some(return_vector)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::error::StateInitializationError;

    #[test]
    fn leap_frog_simulation() -> Result<(), StateInitializationError> {
        let state =
            LatticeStateEFSyncDefault::<LatticeStateDefault<3>, 3>::new_cold(1_f64, 6_f64, 4)?;

        let mut leap_frog = SimulationStateLeap::new_from_state(state.clone());
        assert_eq!(&state, leap_frog.as_ref());

        assert_eq!(
            state.gauss(&LatticePoint::default()),
            leap_frog.gauss(&LatticePoint::default())
        );

        let _: &mut LatticeStateEFSyncDefault<LatticeStateDefault<3>, 3> = leap_frog.as_mut();
        Ok(())
    }
}
