
//! Module containing the simulation

use super::{
    super::{
        field::{
            LinkMatrix,
            EField,
            Su3Adjoint
        },
        integrator::SymplecticIntegrator,
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
    },
    monte_carlo::MonteCarlo,
    SimulationError,
};

use na::{
    Vector3,
    ComplexField,
};
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use crossbeam::thread;


pub trait LatticeState
    where Self: Sync + Sized
{
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix;
    
    fn lattice(&self) -> &LatticeCyclique;
    
    fn beta(&self) -> Real;
    
    const CA: Real;
    
    fn get_hamiltonian_links(&self) -> Real;
    
    /*
    /// Get the probability of `state` to replace `self` for the next element of the Markov chain.
    fn get_probability_of_next_element<State>(&self, state: &State) -> Real
        where State: LatticeState
    {
        (- state.get_hamiltonian_links() + self.get_hamiltonian_links()).exp().min(1_f64)
    }
    */
    
    fn monte_carlo_step<M>(self, m: &mut M) -> Result<Self, SimulationError>
        where M: MonteCarlo<Self>,
    {
        m.get_next_element(self)
    }
}

pub trait LatticeStateNew where Self: LatticeState + Sized {
    ///Create a new simulation state
    fn new(lattice: LatticeCyclique, beta: Real, link_matrix: LinkMatrix) -> Result<Self, SimulationError>;
}

// TODO improve compatibility with other monte carlo algorithm
pub trait LatticeHamiltonianSimulationState
    where Self: Sized + Sync + LatticeState
{
    
    fn reset_e_field(&mut self, rng: &mut impl rand::Rng){
        let new_e_field = EField::new_deterministe(&self.lattice(), rng, &rand_distr::StandardNormal);
        if self.lattice().get_number_of_points() != new_e_field.len() {
            unreachable!()
        }
        *self.e_field_mut() = new_e_field;
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField;
    
    fn  e_field_mut(&mut self) -> &mut EField;
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize;
    
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3>;
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>>;
    
    fn get_hamiltonian_efield(&self) -> Real;
    
    fn get_hamiltonian_total(&self) -> Real {
        self.get_hamiltonian_links() + self.get_hamiltonian_efield()
    }
    
    
}

pub trait LatticeHamiltonianSimulationStateNew where Self: LatticeHamiltonianSimulationState {
    /// Create a new simulation state
    fn new(lattice: LatticeCyclique, beta: Real, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError>;
    
    fn new_random_e(lattice: LatticeCyclique, beta: Real, link_matrix: LinkMatrix, rng: &mut impl rand::Rng) -> Result<Self, SimulationError>
    {
        let e_field = EField::new_deterministe(&lattice, rng, &rand_distr::StandardNormal);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
}

pub trait SimulationStateSynchrone where Self: LatticeHamiltonianSimulationState + Clone {
    fn simulate_to_leapfrog<I, State>(&self, delta_t: Real, integrator: &I) -> Result<State, SimulationError>
        where State: SimulationStateLeapFrog,
        I: SymplecticIntegrator<Self, State>
    {
        integrator.integrate_sync_leap(&self, delta_t)
    }
    
    fn simulate_using_leapfrog_n<I, State>(
        &self,
        delta_t: Real,
        number_of_steps: usize,
        integrator: &I,
    ) -> Result<Self, SimulationError>
        where State: SimulationStateLeapFrog,
        I: SymplecticIntegrator<Self, State>,
    {
        if number_of_steps == 0 {
            return Err(SimulationError::ZeroStep);
        }
        let mut state_leap = self.simulate_to_leapfrog(delta_t, integrator)?;
        if number_of_steps > 1 {
            state_leap = state_leap.simulate_leap_n(delta_t, integrator, number_of_steps -1)?;
        }
        let state_sync = state_leap.simulate_to_synchrone(delta_t, integrator)?;
        Ok(state_sync)
    }
    
    fn simulate_using_leapfrog_n_auto<I>(
        &self,
        delta_t: Real,
        number_of_steps: usize,
        integrator: &I,
    ) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<Self, SimulationStateLeap<Self>>,
    {
        self.simulate_using_leapfrog_n(delta_t, number_of_steps, integrator)
    }
    
    fn simulate_sync<I, T>(&self, delta_t: Real, integrator: &I) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<Self, T>,
        T: SimulationStateLeapFrog,
    {
        integrator.integrate_sync_sync(&self, delta_t)
    }
    
    fn simulate_sync_n<I, T>(&self, delta_t: Real, integrator: &I, numbers_of_times: usize) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<Self, T>,
        T: SimulationStateLeapFrog,
    {
        if numbers_of_times == 0 {
            return Err(SimulationError::ZeroStep);
        }
        let mut state = self.simulate_sync(delta_t, integrator)?;
        for _ in 0..(numbers_of_times - 1) {
            state = state.simulate_sync(delta_t, integrator)?;
        }
        Ok(state)
    }
    
}

pub trait SimulationStateLeapFrog where Self: LatticeHamiltonianSimulationState {
    fn simulate_to_synchrone<I, State>(&self, delta_t: Real, integrator: &I) -> Result<State, SimulationError>
        where State: SimulationStateSynchrone,
        I: SymplecticIntegrator<State, Self>
    {
        integrator.integrate_leap_sync(&self, delta_t)
    }
    
    fn simulate_leap<I, T>(&self, delta_t: Real, integrator: &I) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<T, Self>,
        T: SimulationStateSynchrone,
    {
        integrator.integrate_leap_leap(&self, delta_t)
    }
    
    fn simulate_leap_n<I, T>(&self, delta_t: Real, integrator: &I, numbers_of_times: usize) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<T, Self>,
        T: SimulationStateSynchrone,
    {
        if numbers_of_times == 0 {
            return Err(SimulationError::ZeroStep);
        }
        let mut state = self.simulate_leap(delta_t, integrator)?;
        for _ in 0..(numbers_of_times - 1) {
            state = state.simulate_leap(delta_t, integrator)?;
        }
        Ok(state)
    }
}


/// Represent a simulation state at a set time.
#[derive(Debug, PartialEq, Clone)]
pub struct LatticeStateDefault {
    lattice : LatticeCyclique,
    beta: Real,
    link_matrix: LinkMatrix,
}

impl LatticeStateDefault {
    pub fn new_cold(size: Real, beta: Real , number_of_points: usize) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        Self::new(lattice, beta, link_matrix)
    }
}

impl LatticeStateNew for LatticeStateDefault{
    fn new(lattice: LatticeCyclique, beta: Real, link_matrix: LinkMatrix) -> Result<Self, SimulationError> {
        if lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice, link_matrix, beta})
    }
}

impl LatticeState for LatticeStateDefault{
    const CA: Real = 3_f64;
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        &self.link_matrix
    }
    
    fn lattice(&self) -> &LatticeCyclique {
        &self.lattice
    }
    
    fn beta(&self) -> Real {
        self.beta
    }
    
    /// Get the Hamiltonian of the state.
    fn get_hamiltonian_links(&self) -> Real {
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::POSITIVES_SPACE.iter().map(|dir_i| {
                Direction::POSITIVES_SPACE.iter()
                    .filter(|dir_j| dir_i.to_index() < dir_j.to_index())
                    .map(|dir_j| {
                        1_f64 - self.link_matrix().get_pij(&el, dir_i, dir_j, self.lattice())
                            .unwrap().trace().real() / Self::CA
                    }).sum::<Real>()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
}

/// Represent a simulation state at a set time.
#[derive(Debug, PartialEq, Clone)]
pub struct LatticeHamiltonianSimulationStateSync {
    lattice : LatticeCyclique,
    beta: Real,
    e_field: EField,
    link_matrix: LinkMatrix,
    t: usize,
}

impl SimulationStateSynchrone for LatticeHamiltonianSimulationStateSync{}

impl LatticeHamiltonianSimulationStateSync {
    
    /// Generate a random initial state.
    ///
    /// Single threaded generation with a given random number generator.
    /// `size` is the size parameter of the lattice and `number_of_points` is the number of points
    /// in each spatial dimension of the lattice. See [LatticeCyclique::new] for more info.
    ///
    /// useful to reproduce a set of data but slower than [`LatticeHamiltonianSimulationStateSync::new_random_threaded`]
    /// # Example
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::{simulation::LatticeHamiltonianSimulationStateSync, lattice::LatticeCyclique};
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// assert_eq!(
    ///     LatticeHamiltonianSimulationStateSync::new_deterministe(1_f64, 1_f64, 4, &mut rng_1, &distribution).unwrap(),
    ///     LatticeHamiltonianSimulationStateSync::new_deterministe(1_f64, 1_f64, 4, &mut rng_2, &distribution).unwrap()
    /// );
    /// ```
    pub fn new_deterministe(
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points)
            .ok_or(SimulationError::InitialisationError)?;
        let e_field = EField::new_deterministe(&lattice, rng, d);
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng, d);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
    
    pub fn new_deterministe_cold_e_hot_link (
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points)
            .ok_or(SimulationError::InitialisationError)?;
        let e_field = EField::new_cold(&lattice);
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng, d);
        
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
    
    /// Generate a random initial state.
    ///
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reproduce easily, In that case use
    /// [`LatticeHamiltonianSimulationStateSync::new_deterministe`].
    pub fn new_random_threaded<Distribution>(
        size: Real,
        beta: Real,
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
            return Self::new_deterministe(size, beta, number_of_points, &mut rng, d);
        }
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let result = thread::scope(|s| {
            let lattice_clone = lattice.clone();
            let handel = s.spawn(move |_| {
                EField::new_random(&lattice_clone, d)
            });
            let link_matrix = LinkMatrix::new_random_threaded(&lattice, d, number_of_points - 1)
                .map_err(SimulationError::ThreadingError)?;
            
            let e_field = handel.join().map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
            // not very clean: TODO imporve
            Ok(Self::new(lattice, beta, e_field, link_matrix, 0)?)
        }).map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
        return result;
    }
    
    pub fn new_cold(size: Real, beta: Real , number_of_points: usize) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        let e_field = EField::new_cold(&lattice);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
    
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn get_gauss(&self, point: &LatticePoint) -> Option<CMatrix3> {
        Direction::DIRECTIONS_SPACE.iter().map(|dir| {
            let e_i = self.e_field().get_e_field(point, dir, self.lattice())?;
            let u_mi = self.link_matrix().get_matrix(&LatticeLink::new(*point, - *dir), self.lattice())?;
            let p_mi = self.lattice().add_point_direction(*point, & - *dir);
            let e_m_i = self.e_field().get_e_field(&p_mi, dir, self.lattice())?;
            Some(e_i.to_matrix() - u_mi * e_m_i.to_matrix() * u_mi.adjoint())
        }).sum::<Option<CMatrix3>>()
    }
}

impl LatticeState for LatticeHamiltonianSimulationStateSync {
    
    const CA: Real = 3_f64;
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        &self.link_matrix
    }
    
    fn lattice(&self) -> &LatticeCyclique {
        &self.lattice
    }
    
    fn beta(&self) -> Real {
        self.beta
    }
    
    /// Get the Hamiltonian of the state.
    fn get_hamiltonian_links(&self) -> Real {
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::POSITIVES_SPACE.iter().map(|dir_i| {
                Direction::POSITIVES_SPACE.iter()
                    .filter(|dir_j| dir_i.to_index() < dir_j.to_index())
                    .map(|dir_j| {
                        1_f64 - self.link_matrix().get_pij(&el, dir_i, dir_j, self.lattice())
                            .unwrap().trace().real() / Self::CA
                    }).sum::<Real>()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
}

impl LatticeHamiltonianSimulationStateNew for LatticeHamiltonianSimulationStateSync{
    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    fn new(lattice: LatticeCyclique, beta: Real, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        if lattice.get_number_of_points() != e_field.len() ||
            lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice, e_field, link_matrix, t, beta})
    }
}

impl LatticeHamiltonianSimulationState for LatticeHamiltonianSimulationStateSync {
    
    fn get_hamiltonian_efield(&self) -> Real {
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::POSITIVES_SPACE.iter().map(|dir_i| {
                let e_i = self.e_field().get_e_field(&el, dir_i, self.lattice()).unwrap().to_matrix();
                (e_i * e_i).trace().real()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField {
        &self.e_field
    }
    
    fn e_field_mut(&mut self) -> &mut EField {
        &mut self.e_field
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.t
    }
    
    /// Get the derive of U_i(x).
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * Self::CA ).sqrt();
        let u_i = self.link_matrix().get_matrix(&LatticeLink::from(*link), self.lattice())?;
        let e_i = self.e_field().get_e_field(link.pos(), link.dir(), self.lattice())?;
        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / self.lattice().size()));
    }
    
    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        let c = - (2_f64 / Self::CA).sqrt();
        let mut iterator = Direction::POSITIVES_SPACE.iter().map(|dir| {
            let u_i = self.link_matrix().get_matrix(&LatticeLink::new(*point, *dir), self.lattice())?;
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
    
}

/// Represent a simulation state using leap frog using a synchrone type
#[derive(Debug, PartialEq, Clone)]
pub struct SimulationStateLeap<State>
    where State: SimulationStateSynchrone
{
    state: State
}

impl<State> SimulationStateLeap<State>
    where State: SimulationStateSynchrone + LatticeHamiltonianSimulationState
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
    
    pub fn from_synchrone<I>(s: &State, integrator: &I , delta_t: Real) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<State, Self>
    {
        s.simulate_to_leapfrog(delta_t, integrator)
    }
    
}

impl<State> SimulationStateLeapFrog for SimulationStateLeap<State>
    where State: SimulationStateSynchrone + LatticeHamiltonianSimulationState
{}

impl<State> LatticeState for SimulationStateLeap<State>
    where State: LatticeHamiltonianSimulationState + SimulationStateSynchrone
{
    const CA: Real = State::CA;
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        self.state().link_matrix()
    }
    
    fn lattice(&self) -> &LatticeCyclique {
        self.state().lattice()
    }
    
    fn beta(&self) -> Real {
        self.state().beta()
    }
    
    fn get_hamiltonian_links(&self) -> Real {
        self.state().get_hamiltonian_links()
    }
}

impl<State> LatticeHamiltonianSimulationStateNew for SimulationStateLeap<State>
    where State: LatticeHamiltonianSimulationState + SimulationStateSynchrone + LatticeHamiltonianSimulationStateNew
{
    fn new(lattice: LatticeCyclique, beta: Real, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        let state = State::new(lattice, beta, e_field, link_matrix, t)?;
        Ok(Self {state})
    }
}

impl<State> LatticeHamiltonianSimulationState for SimulationStateLeap<State>
    where State: LatticeHamiltonianSimulationState + SimulationStateSynchrone
{
    
    fn get_hamiltonian_efield(&self) -> Real {
        self.state().get_hamiltonian_efield()
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField {
        self.state().e_field()
    }
    
    fn e_field_mut(&mut self) -> &mut EField {
        self.state.e_field_mut()
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.state().t()
    }
    
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        self.state().get_derivative_e(point)
    }
    
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        self.state().get_derivatives_u(link)
    }
}


#[derive(Debug, PartialEq, Clone)]
pub struct LatticeHamiltonianSimulationStateSyncDefault<State>
    where State: LatticeState
{
    lattice_state: State,
    e_field: EField,
    t: usize,
}

impl<State> LatticeHamiltonianSimulationStateSyncDefault<State>
    where State: LatticeState,
{
    pub fn get_state_owned(self) -> State {
        self.lattice_state
    }
    
    pub fn lattice_state(&self) -> &State {
        &self.lattice_state
    }
    
    pub fn new_random_e_state(lattice_state: State, rng: &mut impl rand::Rng) -> Self {
        let e_field = EField::new_deterministe(&lattice_state.lattice(), rng, &rand_distr::StandardNormal);
        Self{lattice_state, e_field, t: 0}
    }
}


impl<State> SimulationStateSynchrone for LatticeHamiltonianSimulationStateSyncDefault<State>
    where State: LatticeState + Clone
{}

impl<State> LatticeState for LatticeHamiltonianSimulationStateSyncDefault<State>
    where State: LatticeState
{
    fn link_matrix(&self) -> &LinkMatrix{
        self.lattice_state.link_matrix()
    }

    fn lattice(&self) -> &LatticeCyclique {
        self.lattice_state.lattice()
    }

    fn beta(&self) -> Real {
        self.lattice_state.beta()
    }

    const CA: Real = State::CA;

    fn get_hamiltonian_links(&self) -> Real {
        self.lattice_state.get_hamiltonian_links()
    }
}

impl<State> LatticeHamiltonianSimulationStateNew for LatticeHamiltonianSimulationStateSyncDefault<State>
    where State: LatticeState + LatticeStateNew
{
    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    fn new(lattice: LatticeCyclique, beta: Real, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        if lattice.get_number_of_points() != e_field.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice_state: State::new(lattice, beta, link_matrix)?, e_field, t})
    }
}

impl<State> LatticeHamiltonianSimulationState for LatticeHamiltonianSimulationStateSyncDefault<State>
    where State: LatticeState
{
    
    fn get_hamiltonian_efield(&self) -> Real {
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::POSITIVES_SPACE.iter().map(|dir_i| {
                let e_i = self.e_field().get_e_field(&el, dir_i, self.lattice()).unwrap().to_matrix();
                (e_i * e_i).trace().real()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField {
        &self.e_field
    }
    
    fn e_field_mut(&mut self) -> &mut EField {
        &mut self.e_field
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.t
    }
    
    /// Get the derive of U_i(x).
    fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * Self::CA ).sqrt();
        let u_i = self.link_matrix().get_matrix(&LatticeLink::from(*link), self.lattice())?;
        let e_i = self.e_field().get_e_field(link.pos(), link.dir(), self.lattice())?;
        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / self.lattice().size()));
    }
    
    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        let c = - (2_f64 / Self::CA).sqrt();
        let mut iterator = Direction::POSITIVES_SPACE.iter().map(|dir| {
            let u_i = self.link_matrix().get_matrix(&LatticeLink::new(*point, *dir), self.lattice())?;
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
    
}
