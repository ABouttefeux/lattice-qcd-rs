
//! Module containing the simulation PartialEq

use super::{
    field::{
        LinkMatrix,
        EField,
        Su3Adjoint
    },
    integrator::Integrator,
    lattice::{
        LatticeCyclique,
        LatticeLink,
        LatticePoint,
        LatticeLinkCanonical,
        Direction,
    },
    thread::{
        ThreadError
    },
    Real,
    CMatrix3,
    Vector8,
    Complex,
    su3,
};

use na::{
    Vector3,
    ComplexField,
};
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use crossbeam::thread;

/// Error returned by simulation.
#[derive(Debug)]
pub enum SimulationError {
    /// multithreading error, see [`ThreadError`].
    ThreadingError(ThreadError),
    /// Error while initialising.
    InitialisationError,
}

impl From<ThreadError> for SimulationError{
    fn from(err: ThreadError) -> Self{
        SimulationError::ThreadingError(err)
    }
}

pub trait SimulationState
    where Self: Sized
{
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3>;
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>>;
    fn get_hamiltonian(&self) -> Real;
    
    fn simulate<I>(&self, delta_t: Real, integrator: I) -> Result<Self, SimulationError>
        where I: Integrator<Self, Self>
    {
        integrator.integrate(&self, delta_t)
    }
    
}

pub trait SimulationStateSynchrone where Self: SimulationState {
    fn simulate_to_leapfrog<I, State>(&self, delta_t: Real, integrator: I) -> Result<State, SimulationError>
        where State: SimulationStateLeapFrog,
        I: Integrator<Self, State>
    {
        integrator.integrate(&self, delta_t)
    }
}

pub trait SimulationStateLeapFrog where Self: SimulationState {
    fn simulate_to_synchrone<I, State>(&self, delta_t: Real, integrator: I) -> Result<State, SimulationError>
        where State: SimulationStateSynchrone,
        I: Integrator<Self, State>
    {
        integrator.integrate(&self, delta_t)
    }
}

/// Trait for default lattice simulation state
pub trait LatticeSimulationStateDefault
    where Self: Sized + Sync
{
    fn new(lattice: LatticeCyclique, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError>;
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField;
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix;
    
    fn lattice(&self) -> &LatticeCyclique;
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize;
    
    const CA: Real = 3_f64;
    
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    fn get_gauss(&self, point: &LatticePoint) -> Option<CMatrix3> {
        Direction::POSITIVES_SPACE.iter().map(|dir| {
            let e_i = self.e_field().get_e_field(point, dir, self.lattice())?;
            let u_mi = self.link_matrix().get_matrix(&LatticeLink::new(*point, - dir.clone()), self.lattice())?;
            let p_mi = self.lattice().add_point_direction(*point, & - dir.clone());
            let e_m_i = self.e_field().get_e_field(&p_mi, dir, self.lattice())?;
            Some(e_i.to_matrix() - u_mi * e_m_i.to_matrix() * u_mi.adjoint())
        }).sum::<Option<CMatrix3>>()
    }
}

impl<T> SimulationState for T
    where T: LatticeSimulationStateDefault
{
    
    /// Get the derive of U_i(x).
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * Self::CA ).sqrt();
        let u_i = self.link_matrix().get_matrix(&LatticeLink::from(link.clone()), self.lattice())?;
        let e_i = self.e_field().get_e_field(link.pos(), link.dir(), self.lattice())?;
        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / self.lattice().size()));
    }
    
    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        let c = - (2_f64 / Self::CA).sqrt();
        let mut iterator = Direction::POSITIVES_SPACE.iter().map(|dir| {
            let u_i = self.link_matrix().get_matrix(&LatticeLink::new(*point, dir.clone()), self.lattice())?;
            let sum_s: CMatrix3 = Direction::DIRECTIONS_SPACE.iter()
                .filter(|dir_2| dir_2.to_positive() != *dir)
                .map(|dir_2| {
                    self.link_matrix().get_sij(point, dir, dir_2, self.lattice())
                        .map(|el| el.adjoint())
                }).sum::<Option<CMatrix3>>()?;
            Some(Su3Adjoint::new(
                Vector8::<Real>::from_fn(|index, _| {
                    c * (su3::GENERATORS[index] * u_i * sum_s).trace().imaginary() / self.lattice().size()
                })
            ))
        });
        // TODO cleanup
        Some(Vector3::new(iterator.next().unwrap()?, iterator.next().unwrap()?, iterator.next().unwrap()?))
    }
    
    /// Get the Hamiltonian of the state.
    fn get_hamiltonian(&self) -> Real {
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::POSITIVES_SPACE.iter().map(|dir_i| {
                let sum_plaquette = Direction::POSITIVES_SPACE.iter()
                    .filter(|dir_j| dir_i.to_index() < dir_j.to_index())
                    .map(|dir_j| {
                        1_f64 - self.link_matrix().get_pij(&el, dir_i, dir_j, self.lattice())
                            .unwrap().trace().real() / Self::CA
                    }).sum::<Real>();
                // TODO optimize
                let e_i = self.e_field().get_e_field(&el, dir_i, self.lattice()).unwrap().to_matrix();
                let sum_trace_e = (e_i * e_i).trace().real();
                sum_trace_e + sum_plaquette
            }).sum::<Real>()
        }).sum()
    }
}

/// Represent a simulation state at a set time.
#[derive(Debug, PartialEq, Clone)]
pub struct LatticeSimulationStateSync {
    lattice : LatticeCyclique,
    e_field: EField,
    link_matrix: LinkMatrix,
    t: usize,
}

impl LatticeSimulationStateSync {
    
    /// Generate a random initial state.
    ///
    /// Single threaded generation with a given random number generator.
    /// `size` is the size parameter of the lattice and `number_of_points` is the number of points
    /// in each spatial dimension of the lattice. See [LatticeCyclique::new] for more info.
    ///
    /// useful to reproduce a set of data but slower than [`LatticeSimulationStateSyn::new_random_threaded`]
    /// # Example
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::{simulation::LatticeSimulationStateSync, lattice::LatticeCyclique};
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(- 1_f64..1_f64);
    /// assert_eq!(
    ///     LatticeSimulationStateSync::new_deterministe(1_f64, 4, &mut rng_1, &distribution).unwrap(),
    ///     LatticeSimulationStateSync::new_deterministe(1_f64, 4, &mut rng_2, &distribution).unwrap()
    /// );
    /// ```
    pub fn new_deterministe(
        size: Real,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points)
                .ok_or(SimulationError::InitialisationError)?;
        let e_field = EField::new_deterministe(&lattice, rng, d);
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng, d);
        Self::new(lattice, e_field, link_matrix, 0)
    }
    
    /// Generate a random initial state.
    ///
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reproduce easily, In that case use [`LatticeSimulationStateSyn::new_deterministe`].
    pub fn new_random_threaded<Distribution>(
        size: Real,
        number_of_points: usize,
        d: &Distribution,
        number_of_thread : usize
    ) -> Result<Self, SimulationError>
        where Distribution: rand_distr::Distribution<Real> + Sync,
    {
        if number_of_thread == 0 {
            return Err(SimulationError::ThreadingError(ThreadError::ThreadNumberIncorect));
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return Self::new_deterministe(size, number_of_points, &mut rng, d);
        }
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let result = thread::scope(|s| {
            let lattice_clone = lattice.clone();
            let handel = s.spawn(move |_| {
                EField::new_random(&lattice_clone, d)
            });
            let link_matrix = LinkMatrix::new_random_threaded(&lattice, d, number_of_points - 1)
                .map_err(|err| SimulationError::ThreadingError(err))?;
            
            let e_field = handel.join().map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
            // not very clean: TODO imporve
            Ok(Self::new(lattice, e_field, link_matrix, 0)?)
        }).map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
        return result;
    }
    
    pub fn new_cold(size: Real, number_of_points: usize) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        let e_field = EField::new_cold(&lattice);
        Self::new(lattice, e_field, link_matrix, 0)
    }
}

impl LatticeSimulationStateDefault for LatticeSimulationStateSync {
    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    fn new(lattice: LatticeCyclique, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        if lattice.get_number_of_points() != e_field.len() ||
            lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice, e_field, link_matrix, t})
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField {
        &self.e_field
    }
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        &self.link_matrix
    }
    
    fn lattice(&self) -> &LatticeCyclique {
        &self.lattice
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.t
    }
}

/// Represent a simulation state using leap frog using a synchrone type
#[derive(Debug, PartialEq, Clone)]
pub struct SimulationStateLeap<State>
    where State: SimulationStateSynchrone
{
    state: State
}

impl<State> SimulationStateLeap<State>
    where State: SimulationStateSynchrone + LatticeSimulationStateDefault
    // TODO review this part ` + LatticeSimulationStateDefault`
    // It is require because it should implement
    // I: Integrator<State, Self>
    // so self need to be a SimulationState
    // but this imple is remove because it conflict with the impl of
    // impl<State> LatticeSimulationStateDefault for SimulationStateLeap<State>
    //    where State: LatticeSimulationStateDefault + SimulationStateSynchrone
{
    fn state(&self) -> &State {
        &self.state
    }
    
    pub fn from_synchrone<I>(s: &State, integrator: I , delta_t: Real) -> Result<Self, SimulationError>
        where I: Integrator<State, Self>
    {
        s.simulate_to_leapfrog(delta_t, integrator)
    }
}

/*
// TODO Solve conflicint implementation
impl<State> SimulationState for SimulationStateLeap<State>
    where State: SimulationStateSynchrone
{
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        self.state().get_derivatives_u(link)
    }
    
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        self.state().get_derivative_e(point)
    }
    fn get_hamiltonian(&self) -> Real {
        self.state().get_hamiltonian()
    }
}
*/

impl<State> SimulationStateLeapFrog for SimulationStateLeap<State>
    where State: SimulationStateSynchrone + LatticeSimulationStateDefault
{}

impl<State> LatticeSimulationStateDefault for SimulationStateLeap<State>
    where State: LatticeSimulationStateDefault + SimulationStateSynchrone
{
    fn new(lattice: LatticeCyclique, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        let state = State::new(lattice, e_field, link_matrix, t)?;
        Ok(Self {state})
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField {
        self.state().e_field()
    }
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        self.state().link_matrix()
    }
    
    fn lattice(&self) -> &LatticeCyclique {
        self.state().lattice()
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.state().t()
    }
}
