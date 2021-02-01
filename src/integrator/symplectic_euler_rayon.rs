
//! Basic symplectic Euler integrator using Rayon, slightly faster than [`SymplecticEuler`]


/// Basic symplectic Euler integrator using Rayon, slightly faster than [`SymplecticEuler`]


use super::{
    super::{
        field::{
            EField,
            LinkMatrix,
        },
        thread::{
            run_pool_parallel_rayon,
        },
        Complex,
        lattice::{
            LatticeLink,
            LatticeLinkCanonical,
            LatticePoint,
        },
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

pub struct SymplecticEulerRayon {}

impl SymplecticEulerRayon {
    /// Create a new SymplecticEulerRayon
    pub const fn new() -> Self {
        Self{}
    }
}

impl Default for SymplecticEulerRayon {
    /// Identical to [`SymplecticEulerRayon::new`].
    fn default() -> Self{
        Self::new()
    }
}

impl<State> SymplecticIntegrator<State, State> for SymplecticEulerRayon
    where State: LatticeSimulationStateDefault
{}

impl<State> Integrator<State, State> for  SymplecticEulerRayon
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
        
        let link_matrix = run_pool_parallel_rayon(
            l.lattice().get_links_space(),
            l,
            |link, l| integrate_link_closure(link, l, delta_t),
        );
        
        let e_field = run_pool_parallel_rayon(
            l.lattice().get_points(),
            l,
            |point, l| integrate_efield_closure(point, l, delta_t),
        );
        
        State::new(l.lattice().clone(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
}

// Basic symplectic Euler integrator using Rayon, slightly faster than [`SymplecticEuler`]
pub struct SymplecticEulerRayonToLeap {}

impl SymplecticEulerRayonToLeap {
    /// Create a new SymplecticEulerRayon
    pub const fn new() -> Self {
        Self{}
    }
}

impl Default for SymplecticEulerRayonToLeap {
    /// Identical to [`SymplecticEulerRayon::new`].
    fn default() -> Self{
        Self::new()
    }
}

impl<State1, State2> SymplecticIntegrator<State1, State2> for SymplecticEulerRayonToLeap
where State1: LatticeSimulationStateDefault + SimulationStateSynchrone,
State2: LatticeSimulationStateDefault + SimulationStateLeapFrog
{}

impl<State1, State2> Integrator<State1, State2> for  SymplecticEulerRayonToLeap
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
        
        let e_field = run_pool_parallel_rayon(
            l.lattice().get_points(),
            l,
            |point, l| integrate_efield_closure(point, l, delta_t),
        );
        
        State2::new(l.lattice().clone(), EField::new(e_field), l.link_matrix().clone(), l.t())
    }
}

// Basic symplectic Euler integrator using Rayon, slightly faster than [`SymplecticEuler`]
pub struct SymplecticEulerRayonToSync {}

impl SymplecticEulerRayonToSync {
    /// Create a new SymplecticEulerRayon
    pub const fn new() -> Self {
        Self{}
    }
}

impl Default for SymplecticEulerRayonToSync {
    /// Identical to [`SymplecticEulerRayon::new`].
    fn default() -> Self{
        Self::new()
    }
}

impl<State1, State2> SymplecticIntegrator<State1, State2> for SymplecticEulerRayonToSync
where State1: LatticeSimulationStateDefault + SimulationStateLeapFrog,
State2: LatticeSimulationStateDefault + SimulationStateSynchrone
{}

impl<State1, State2> Integrator<State1, State2> for  SymplecticEulerRayonToSync
    where State1: LatticeSimulationStateDefault + SimulationStateLeapFrog,
    State2: LatticeSimulationStateDefault + SimulationStateSynchrone
{
    fn integrate(&self, l: &State1, delta_t: Real) ->  Result<State2, SimulationError> {
        
        // closure for link intregration
        let integrate_link_closure = |link: &LatticeLinkCanonical, l: &State1, delta_t| {
            let canonical_link = LatticeLink::from(*link);
            let initial_value = l.link_matrix().get_matrix(&canonical_link, l.lattice()).unwrap();
            return initial_value + l.get_derivatives_u(link).unwrap() * Complex::from(delta_t);
        };
        
        // closure for "Electrical" field intregration
        let integrate_efield_closure = |point: &LatticePoint, l: &State1, delta_t| {
            let initial_value = *l.e_field().get_e_vec(point, l.lattice()).unwrap();
            let deriv = l.get_derivative_e(point).unwrap();
            return initial_value + deriv.map(|el| el * delta_t / 2_f64);
        };
        
        let link_matrix = run_pool_parallel_rayon(
            l.lattice().get_links_space(),
            l,
            |link, l| integrate_link_closure(link, l, delta_t),
        );
        
        let e_field = run_pool_parallel_rayon(
            l.lattice().get_points(),
            l,
            |point, l| integrate_efield_closure(point, l, delta_t),
        );
        
        // we advace the counter by one
        State2::new(l.lattice().clone(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1)
    }
}
