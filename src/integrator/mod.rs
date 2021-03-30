
//! Numerical integrators to carry out simulations.
//!
//! See [`SymplecticIntegrator`]. The simulations are done on [`LatticeHamiltonianSimulationState`]
//! It also require a notion of [`SimulationStateSynchrone`] and [`SimulationStateLeapFrog`].
//!
//! Even thought it is effortless to implement both [`SimulationStateSynchrone`] and [`SimulationStateLeapFrog`].
//! I adivce against it and implement only [`SimulationStateSynchrone`] and use [`super::simulation::SimulationStateLeap`]
//! for leap frog states as it gives a compile time check that you did not forget doing a demi steps.
//!
//! This library gives two implementations of [`SymplecticIntegrator`]: [`SymplecticEuler`] and [`SymplecticEulerRayon`].
//! I would advice using [`SymplecticEulerRayon`] if you do not mind the little more momory it uses.
//! # Example
//! let us create a basic random state and let us simulate.
//! ```
//! extern crate rand;
//! extern crate rand_distr;
//! use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationStateSyncDefault;
//! use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationState;
//! use lattice_qcd_rs::simulation::SimulationStateSynchrone;
//! use lattice_qcd_rs::simulation::LatticeStateDefault;
//! use lattice_qcd_rs::integrator::SymplecticEuler;
//! use lattice_qcd_rs::dim::U3;
//!
//! let mut rng = rand::thread_rng();
//! let distribution = rand::distributions::Uniform::from(
//!     -std::f64::consts::PI..std::f64::consts::PI
//! );
//! let state1 = LatticeHamiltonianSimulationStateSyncDefault::new_random_e_state(LatticeStateDefault::<U3>::new_deterministe(100_f64, 1_f64, 4, &mut rng).unwrap(), &mut rng);
//! let state2 = state1.simulate_sync(&SymplecticEuler::new(8), 0.0001_f64).unwrap();
//! let state3 = state2.simulate_sync(&SymplecticEuler::new(8), 0.0001_f64).unwrap();
//! ```
//! Let us then compute and compare the Hamiltonian.
//! ```
//! # extern crate rand;
//! # extern crate rand_distr;
//! # use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationStateSyncDefault;
//! # use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationState;
//! # use lattice_qcd_rs::simulation::SimulationStateSynchrone;
//! # use lattice_qcd_rs::simulation::LatticeStateDefault;
//! # use lattice_qcd_rs::integrator::SymplecticEuler;
//! # use lattice_qcd_rs::dim::U3;
//! #
//! # let mut rng = rand::thread_rng();
//! # let distribution = rand::distributions::Uniform::from(
//! #    -std::f64::consts::PI..std::f64::consts::PI
//! # );
//! # let state1 = LatticeHamiltonianSimulationStateSyncDefault::new_random_e_state(LatticeStateDefault::<U3>::new_deterministe(100_f64, 1_f64, 4, &mut rng).unwrap(), &mut rng);
//! # let state2 = state1.simulate_sync(&SymplecticEuler::new(8), 0.0001_f64).unwrap();
//! # let state3 = state2.simulate_sync(&SymplecticEuler::new(8), 0.0001_f64).unwrap();
//! let h = state1.get_hamiltonian_total();
//! let h2 = state3.get_hamiltonian_total();
//! println!("The error on the Hamiltonian is {}", h - h2);
//! ```

use super::{
    simulation::{
        SimulationError,
        LatticeHamiltonianSimulationState,
        SimulationStateLeapFrog,
        SimulationStateSynchrone,
    },
    Real,
    Complex,
    lattice::{
        LatticeLinkCanonical,
        LatticeLink,
        LatticePoint,
        LatticeCyclique,
        Direction,
        DirectionList,
    },
    CMatrix3,
    field::{
        Su3Adjoint,
        LinkMatrix,
        EField,
    },
};
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
};

pub mod symplectic_euler;
pub mod symplectic_euler_rayon;

pub use symplectic_euler::SymplecticEuler;
pub use symplectic_euler_rayon::SymplecticEulerRayon;

/*
/// Define an numerical integrator
pub trait Integrator<State, State2>
    where State: LatticeHamiltonianSimulationState,
    State2: LatticeHamiltonianSimulationState,
{
    /// Do one simulation step
    fn integrate(&self, l: &State, delta_t: Real) -> Result<State2, SimulationError>;
}
*/

/// Define an symplectic numerical integrator
///
/// The integrator evlove the state in time.
///
/// The integrator should be capable of switching between Sync state
/// (q (or link matrices) at time T , p (or e_field) at time T )
/// and leap frog (a at time T, p at time T + 1/2)
pub trait SymplecticIntegrator<StateSync, StateLeap, D>
    where StateSync: SimulationStateSynchrone<D>,
    StateLeap: SimulationStateLeapFrog<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    //TODO error
    
    /// Integrate a sync state to a sync state.
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_sync_sync(&self, l: &StateSync, delta_t: Real) -> Result<StateSync, SimulationError>;
    
    /// Integrate a leap state to a leap state using leap frog algorithm.
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_leap_leap(&self, l: &StateLeap, delta_t: Real) -> Result<StateLeap, SimulationError>;
    
    /// Integrate a sync state to a leap state by doing a half step for the conjugate momenta.
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_sync_leap(&self, l: &StateSync, delta_t: Real) -> Result<StateLeap, SimulationError>;
    
    /// Integrate a leap state to a sync state by finishing doing a step for the position and finishing
    /// the half step for the conjugate momenta.
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_leap_sync(&self, l: &StateLeap, delta_t: Real) -> Result<StateSync, SimulationError>;
    
    /// Integrate a Sync state by going to leap and then back to sync.
    /// This is the symplectic methode of integration, which should conserve the hamiltonian
    ///
    /// # Errors
    /// Return an error if the integration encounter a problem
    fn integrate_symplectic(&self, l: &StateSync, delta_t: Real) -> Result<StateSync, SimulationError> {
        self.integrate_leap_sync(&self.integrate_sync_leap(l, delta_t)?, delta_t)
    }
}

/// function for link intregration
fn integrate_link<State, D>(link: &LatticeLinkCanonical<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>, delta_t: Real) -> CMatrix3
    where State: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    let canonical_link = LatticeLink::from(*link);
    let initial_value = link_matrix.get_matrix(&canonical_link, lattice).expect("Link matrix not found");
    initial_value + State::get_derivative_u(link, link_matrix, e_field, lattice).expect("Derivative not found") * Complex::from(delta_t)
}

/// function for "Electrical" field intregration
fn integrate_efield<State, D>(point: &LatticePoint<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>, delta_t: Real) -> VectorN<Su3Adjoint, D>
    where State: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    let initial_value = e_field.get_e_vec(point, lattice).expect("E Field not found");
    let deriv = State::get_derivative_e(point, link_matrix, e_field, lattice).expect("Derivative not found");
    initial_value + deriv.map(|el| el * delta_t)
}
