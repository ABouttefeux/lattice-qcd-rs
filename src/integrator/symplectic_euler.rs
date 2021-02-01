
//! Basic symplectic Euler integrator

use na::{
    Vector3,
};
use super::{
    super::{
        field::{
            Su3Adjoint,
            EField,
            LinkMatrix,
        },
        thread::{
            run_pool_parallel_vec,
        },
        Complex,
        lattice::{
            LatticeLink,
            LatticeLinkCanonical,
            LatticePoint,
        },
        CMatrix3,
        Real,
        simulation::{
            SimulationError,
            SimulationState,
            LatticeSimulationStateDefault,
            SimulationStateSynchrone,
            SimulationStateLeapFrog,
        },
    },
    Integrator,
    SymplecticIntegrator,
};

/// Basic symplectic Euler integrator
pub struct SymplecticEuler
{
    number_of_thread: usize,
}

impl SymplecticEuler {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
    
}

impl Default for SymplecticEuler {
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State> SymplecticIntegrator<State, State> for SymplecticEuler
    where State: LatticeSimulationStateDefault
{}


impl<State> Integrator<State, State> for  SymplecticEuler
    where State: LatticeSimulationStateDefault
{
    fn integrate(&self, l: &State, delta_t: Real) ->  Result<State, SimulationError> {
        
        // closure for link intregration
        let integrate_link_closure = |link: &LatticeLinkCanonical, l: &State, delta_t| {
            let canonical_link = LatticeLink::from(*link);
            let initial_value = l.link_matrix().get_matrix(&canonical_link, l.lattice()).unwrap();
            return initial_value + l.get_derivatives_u(link).unwrap() * Complex::from(delta_t);
        };
        
        // closure for "Electrical" field intregration
        let integrate_efield_closure = |point: &LatticePoint, l: &State, delta_t| {
            let initial_value = *l.e_field().get_e_vec(point, l.lattice()).unwrap();
            let deriv = l.get_derivative_e(point).unwrap();
            return initial_value + deriv.map(|el| el * delta_t);
        };
        
        let number_of_thread = self.number_of_thread;
        let link_matrix = run_pool_parallel_vec(
            l.lattice().get_links_space(),
            l,
            &|link, l| integrate_link_closure(link, l, delta_t),
            number_of_thread,
            l.lattice().get_number_of_canonical_links_space(),
            l.lattice(),
            CMatrix3::zeros(),
        )?;
        
        let e_field = run_pool_parallel_vec(
            l.lattice().get_points(),
            l,
            &|point, l| integrate_efield_closure(point, l, delta_t),
            number_of_thread,
            l.lattice().get_number_of_points(),
            l.lattice(),
            Vector3::from_element(Su3Adjoint::default()),
        )?;
        
        State::new(l.lattice().clone(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
}

/// Basic symplectic Euler integrator used for switching between syn and leapfrog
pub struct SymplecticEulerToLeap
{
    number_of_thread: usize,
}

impl SymplecticEulerToLeap {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
    
}

impl Default for SymplecticEulerToLeap{
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State1, State2> SymplecticIntegrator<State1, State2> for SymplecticEulerToLeap
where State1: LatticeSimulationStateDefault + SimulationStateSynchrone,
State2: LatticeSimulationStateDefault + SimulationStateLeapFrog
{}

impl<State1, State2> Integrator<State1, State2> for  SymplecticEulerToLeap
    where State1: LatticeSimulationStateDefault + SimulationStateSynchrone,
    State2: LatticeSimulationStateDefault + SimulationStateLeapFrog
{
    fn integrate(&self, l: &State1, delta_t: Real) ->  Result<State2, SimulationError> {
        
        // closure for "Electrical" field intregration
        let integrate_efield_closure = |point: &LatticePoint, l: &State1, delta_t| {
            let initial_value = *l.e_field().get_e_vec(point, l.lattice()).unwrap();
            let deriv = l.get_derivative_e(point).unwrap();
            return initial_value + deriv.map(|el| el * delta_t / 2_f64);
        };
        
        let number_of_thread = self.number_of_thread;
        let e_field = run_pool_parallel_vec(
            l.lattice().get_points(),
            l,
            &|point, l| integrate_efield_closure(point, l, delta_t),
            number_of_thread,
            l.lattice().get_number_of_points(),
            l.lattice(),
            Vector3::from_element(Su3Adjoint::default()),
        )?;
        // we do not advance the step counter
        State2::new(l.lattice().clone(), EField::new(e_field), l.link_matrix().clone(), l.t())
    }
}

/// Basic symplectic Euler integrator used for switching between syn and leapfrog
pub struct SymplecticEulerToSynch
{
    number_of_thread: usize,
}

impl SymplecticEulerToSynch {
    /// Create a integrator using a set number of threads
    pub const fn new(number_of_thread: usize) -> Self {
        Self {number_of_thread}
    }
    
}

impl Default for SymplecticEulerToSynch{
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
}

impl<State1, State2> SymplecticIntegrator<State1, State2> for SymplecticEulerToSynch
where State1: LatticeSimulationStateDefault + SimulationStateLeapFrog,
State2: LatticeSimulationStateDefault + SimulationStateSynchrone
{}

impl<State1, State2> Integrator<State1, State2> for  SymplecticEulerToSynch
    where State1: LatticeSimulationStateDefault + SimulationStateLeapFrog,
    State2: LatticeSimulationStateDefault + SimulationStateSynchrone
{
    fn integrate(&self, l: &State1, delta_t: Real) ->  Result<State2, SimulationError> {
        
        let integrate_link_closure = |link: &LatticeLinkCanonical, l: &State1, delta_t| {
            let canonical_link = LatticeLink::from(*link);
            let initial_value = l.link_matrix().get_matrix(&canonical_link, l.lattice()).unwrap();
            return initial_value + l.get_derivatives_u(link).unwrap() * Complex::from(delta_t);
        };
        
        // closure for "Electrical" field intregration
        let integrate_efield_closure = |point: &LatticePoint, l: &State1, delta_t| {
            let initial_value = *l.e_field().get_e_vec(point, l.lattice()).unwrap();
            let deriv = l.get_derivative_e(point).unwrap();
            // we do half a step in the velocity direction to go back to synchrone data
            return initial_value + deriv.map(|el| el * delta_t / 2.0);
        };
        
        let number_of_thread = self.number_of_thread;
        let link_matrix = run_pool_parallel_vec(
            l.lattice().get_links_space(),
            l,
            &|link, l| integrate_link_closure(link, l, delta_t),
            number_of_thread,
            l.lattice().get_number_of_canonical_links_space(),
            l.lattice(),
            CMatrix3::zeros(),
        )?;
        
        let number_of_thread = self.number_of_thread;
        let e_field = run_pool_parallel_vec(
            l.lattice().get_points(),
            l,
            &|point, l| integrate_efield_closure(point, l, delta_t),
            number_of_thread,
            l.lattice().get_number_of_points(),
            l.lattice(),
            Vector3::from_element(Su3Adjoint::default()),
        )?;
        // we do not advance the step counter
        State2::new(l.lattice().clone(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
}
