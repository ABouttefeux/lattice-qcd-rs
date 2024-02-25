//! Numerical integrators to carry out simulations.
//!
//! See [`SymplecticIntegrator`]. The simulations are done on [`LatticeStateWithEField`]
//! It also require a notion of [`SimulationStateSynchronous`]
//! and [`SimulationStateLeapFrog`].
//!
//! Even thought it is effortless to implement both [`SimulationStateSynchronous`]
//! and [`SimulationStateLeapFrog`].
//! I advice against it and implement only [`SimulationStateSynchronous`] and
//! use [`super::simulation::SimulationStateLeap`]
//! for leap frog states as it gives a compile time check that you did not forget
//! doing a half steps.
//!
//! This library gives two implementations of [`SymplecticIntegrator`]:
//! [`SymplecticEuler`] and [`SymplecticEulerRayon`].
//! I would advice using [`SymplecticEulerRayon`] if you do not mind the little
//! more memory it uses.
//! # Example
//! let us create a basic random state and let us simulate.
//! ```
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use lattice_qcd_rs::integrator::{SymplecticEuler, SymplecticIntegrator};
//! use lattice_qcd_rs::simulation::{
//!     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField,
//! };
//! use rand::SeedableRng;
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
//! let state1 = LatticeStateEFSyncDefault::new_random_e_state(
//!     LatticeStateDefault::<3>::new_determinist(100_f64, 1_f64, 4, &mut rng)?,
//!     &mut rng,
//! );
//! let integrator = SymplecticEuler::default();
//! let state2 = integrator.integrate_sync_sync(&state1, 0.000_1_f64)?;
//! let state3 = integrator.integrate_sync_sync(&state2, 0.000_1_f64)?;
//! #     Ok(())
//! # }
//! ```
//! Let us then compute and compare the Hamiltonian.
//! ```
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! # use lattice_qcd_rs::integrator::{SymplecticEuler, SymplecticIntegrator};
//! # use lattice_qcd_rs::simulation::{
//! #     LatticeStateDefault, LatticeStateWithEField, LatticeStateEFSyncDefault,
//! # };
//! # use rand::SeedableRng;
//! #
//! # let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
//! # let state1 = LatticeStateEFSyncDefault::new_random_e_state(
//! #     LatticeStateDefault::<3>::new_determinist(100_f64, 1_f64, 4, &mut rng)?,
//! #     &mut rng,
//! # );
//! # let integrator = SymplecticEuler::default();
//! # let state2 = integrator.integrate_sync_sync(&state1, 0.000_1_f64)?;
//! # let state3 = integrator.integrate_sync_sync(&state2, 0.000_1_f64)?;
//! let h = state1.hamiltonian_total();
//! let h2 = state3.hamiltonian_total();
//! println!("The error on the Hamiltonian is {}", h - h2);
//! #     Ok(())
//! # }
//! ```
//! See [`SimulationStateSynchronous`] for more convenient methods.

use nalgebra::SVector;

use super::{
    field::{EField, LinkMatrix, Su3Adjoint},
    lattice::{LatticeCyclic, LatticeLink, LatticeLinkCanonical, LatticePoint},
    simulation::{LatticeStateWithEField, SimulationStateLeapFrog, SimulationStateSynchronous},
    CMatrix3, Complex, Real,
};

pub mod symplectic_euler;
pub mod symplectic_euler_rayon;

pub use symplectic_euler::SymplecticEuler;
pub use symplectic_euler_rayon::SymplecticEulerRayon;

/// Define an symplectic numerical integrator
///
/// The integrator evolve the state in time.
///
/// The integrator should be capable of switching between Sync state
/// (`q` (or link matrices) at time `T` , `p` (or `e_field`) at time `T` )
/// and leap frog (a at time `T`, `p` at time `T + 1/2`)
///
/// # Example
/// For an example see the module level documentation [`super::integrator`].
pub trait SymplecticIntegrator<StateSync, StateLeap, const D: usize>
where
    StateSync: SimulationStateSynchronous<D>,
    StateLeap: SimulationStateLeapFrog<D>,
{
    /// Type of error returned by the Integrator.
    type Error;

    /// Integrate a sync state to a sync state by advancing the link matrices and the electrical field simultaneously.
    ///
    /// # Example
    /// see the module level documentation [`super::integrator`].
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_sync_sync(&self, l: &StateSync, delta_t: Real) -> Result<StateSync, Self::Error>;

    /// Integrate a leap state to a leap state using leap frog algorithm.
    ///
    ///
    /// # Example
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use lattice_qcd_rs::integrator::{SymplecticEulerRayon, SymplecticIntegrator};
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField,
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
    /// let mut leap_frog = integrator.integrate_sync_leap(&state, 0.000_001_f64)?;
    /// drop(state);
    /// for _ in 0..2 {
    ///     // Realistically you would want more steps
    ///     leap_frog = integrator.integrate_leap_leap(&leap_frog, 0.000_001_f64)?;
    /// }
    /// let state = integrator.integrate_leap_sync(&leap_frog, 0.000_001_f64)?;
    /// let h2 = state.hamiltonian_total();
    ///
    /// println!("The error on the Hamiltonian is {}", h - h2);
    /// #     Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_leap_leap(&self, l: &StateLeap, delta_t: Real) -> Result<StateLeap, Self::Error>;

    /// Integrate a sync state to a leap state by doing a half step for the conjugate momenta.
    ///
    /// # Example
    /// see [`SymplecticIntegrator::integrate_leap_leap`]
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_sync_leap(&self, l: &StateSync, delta_t: Real) -> Result<StateLeap, Self::Error>;

    /// Integrate a leap state to a sync state by finishing doing a step for the position and finishing
    /// the half step for the conjugate momenta.
    ///
    ///  # Example
    /// see [`SymplecticIntegrator::integrate_leap_leap`]
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_leap_sync(&self, l: &StateLeap, delta_t: Real) -> Result<StateSync, Self::Error>;

    /// Integrate a Sync state by going to leap and then back to sync.
    /// This is the symplectic method of integration, which should conserve approximately the hamiltonian
    ///
    /// Note that you might want to override this method as it can save you from a clone.
    ///
    /// # Example
    /// ```
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use lattice_qcd_rs::integrator::{SymplecticEulerRayon, SymplecticIntegrator};
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField,
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
    ///     state = integrator.integrate_symplectic(&state, 0.000_001_f64)?;
    /// }
    /// let h2 = state.hamiltonian_total();
    ///
    /// println!("The error on the Hamiltonian is {}", h - h2);
    /// #     Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    #[inline]
    fn integrate_symplectic(&self, l: &StateSync, delta_t: Real) -> Result<StateSync, Self::Error> {
        self.integrate_leap_sync(&self.integrate_sync_leap(l, delta_t)?, delta_t)
    }
}

/// function for link integration.
/// This must succeed as it is use while doing parallel computation. Returning a Option is undesirable.
/// As it can panic if a out of bound link is passed it needs to stay private.
///
/// # Panic
/// It panics if a out of bound link is passed.
fn integrate_link<State, const D: usize>(
    link: &LatticeLinkCanonical<D>,
    link_matrix: &LinkMatrix,
    e_field: &EField<D>,
    lattice: &LatticeCyclic<D>,
    delta_t: Real,
) -> CMatrix3
where
    State: LatticeStateWithEField<D>,
{
    let canonical_link = LatticeLink::from(*link);
    let initial_value = link_matrix
        .matrix(&canonical_link, lattice)
        .expect("Link matrix not found");
    initial_value
        + State::derivative_u(link, link_matrix, e_field, lattice).expect("Derivative not found")
            * Complex::from(delta_t)
}

/// function for "Electrical" field integration.
/// Like [`integrate_link`] this must succeed.
///
/// # Panics
/// It panics if a out of bound link is passed.
fn integrate_efield<State, const D: usize>(
    point: &LatticePoint<D>,
    link_matrix: &LinkMatrix,
    e_field: &EField<D>,
    lattice: &LatticeCyclic<D>,
    delta_t: Real,
) -> SVector<Su3Adjoint, D>
where
    State: LatticeStateWithEField<D>,
{
    let initial_value = e_field.e_vec(point, lattice).expect("E Field not found");
    let deriv =
        State::derivative_e(point, link_matrix, e_field, lattice).expect("Derivative not found");
    initial_value + deriv.map(|el| el * delta_t)
}
