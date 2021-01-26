
use na::{
    Vector3,
};
use super::{
    field::{
        LatticeSimulationState,
        Su3Adjoint,
        EField,
        LinkMatrix,
    },
    thread::{
        run_pool_parallel_vec,
        ThreadError,
    },
    Complex,
    lattice::{
        LatticeLink,
    },
    CMatrix3,
};

pub trait Integrator {
    fn integrate(l: &LatticeSimulationState, delta_t: f64, number_of_thread: usize) ->  Result<LatticeSimulationState, ThreadError>;
}
pub trait SymplecticIntegrator where Self:Integrator {}

pub struct EulerIntegrator {}

impl EulerIntegrator {
    pub fn new() -> Self {
        Self{}
    }
}

impl Default for EulerIntegrator{
    fn default() -> Self {
        EulerIntegrator::new()
    }
}

impl SymplecticIntegrator for EulerIntegrator{}

impl Integrator for  EulerIntegrator {
    fn integrate(l: &LatticeSimulationState, delta_t: f64, number_of_thread: usize) ->  Result<LatticeSimulationState, ThreadError> {
        let link_matrix = run_pool_parallel_vec(
            l.lattice().get_links_space(l.t()),
            l,
            &|link, l| {
                let canonical_link = LatticeLink::from(link.clone());
                let initial_value = l.link_matrix().get_matrix(l.lattice(), &canonical_link).unwrap();
                return initial_value + l.get_derivatives_u(link).unwrap() * Complex::from(delta_t);
            },
            number_of_thread,
            l.lattice().get_number_of_canonical_links_space(),
            l.lattice(),
            CMatrix3::zeros(),
        )?;
        
        let e_field = run_pool_parallel_vec(
            l.lattice().get_points(l.t()),
            l,
            &|point, l| {
                let mut initial_value = l.e_field().get_e_vec(point, l.lattice()).unwrap().clone();
                let deriv = l.get_derivative_e(point).unwrap();
                for i in 0..initial_value.len() {
                    initial_value[i] = initial_value[i] + deriv[i] * delta_t;
                }
                return initial_value;
            },
            number_of_thread,
            l.lattice().get_number_of_points(),
            l.lattice(),
            Vector3::from_element(Su3Adjoint::default()),
        )?;
        
        Ok(LatticeSimulationState::new(l.lattice().clone(), EField::new(e_field), LinkMatrix::new(link_matrix), l.t() + 1))
    }
}
