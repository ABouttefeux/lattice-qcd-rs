
//! Module containing the simulation State

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
            LatticeElementToIndex,
            DirectionList,
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
    ComplexField,
    Vector4,
    VectorN,
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
};
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use crossbeam::thread;
use std::marker::PhantomData;

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};


/// trait to represent a pure gauge lattice state.
///
/// It defines only one field link_matrix.
pub trait LatticeState<D>
    where Self: Sync + Sized + core::fmt::Debug,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix;
    
    /// Replace the links matrices with the given input. It should panic if link matrix is not of the correct size.
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_canonical_links_space()`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix);
    
    /// get the lattice into which the state exists
    fn lattice(&self) -> &LatticeCyclique<D>;
    
    /// return the beta parameter of the states
    fn beta(&self) -> Real;
    
    /// C_A constant of the model
    const CA: Real;
    
    /// return the Hamiltonian of the links configuration
    fn get_hamiltonian_links(&self) -> Real;
    
    /// Do one monte carlo step with the given
    fn monte_carlo_step<M>(self, m: &mut M) -> Result<Self, SimulationError>
        where M: MonteCarlo<Self, D>,
    {
        m.get_next_element(self)
    }
    
    /// Take the average of the trace of all plaquettes
    fn average_trace_plaquette(&self) -> Option<Complex> {
        self.link_matrix().average_trace_plaquette(self.lattice())
    }
    
}

/// trait for a way to create a [`LatticeState`] from some parameters.
///
/// It is separated from the [`LatticeState`] because not all [`LatticeState`] can be create in this way.
/// By instance when there is also a field of conjugate momenta of the link matrices.
pub trait LatticeStateNew<D>
    where Self: LatticeState<D> + Sized,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    ///Create a new simulation state
    fn new(lattice: LatticeCyclique<D>, beta: Real, link_matrix: LinkMatrix) -> Result<Self, SimulationError>;
}

/// Represent a lattice state where the conjugate momenta of the link matrices are included.
///
/// If you have a LatticeState and want the default way of adding the conjugate momenta look at
/// [`LatticeHamiltonianSimulationStateSyncDefault`].
///
/// If you want to solve the equation of motion using an [`SymplecticIntegrator`] also implement
/// [`SimulationStateSynchrone`] and [`SimulationStateLeap`] can give you an [`SimulationStateLeapFrog`].
///
/// It is used for the [`super::monte_carlo::HybridMonteCarlo`] algorithm.
pub trait LatticeHamiltonianSimulationState<D>
    where Self: Sized + Sync + LatticeState<D> + core::fmt::Debug,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// Reset the e_field with radom value distributed as N(0, 1/beta ) [`rand_distr::StandardNormal`].
    /// # Panic
    /// Panics if N(0, 0.5/beta ) is not a valide distribution (for exampple beta = 0)
    fn reset_e_field(&mut self, rng: &mut impl rand::Rng){
        // &rand_distr::StandardNormal
        // TODO verify
        let d = rand_distr::Normal::new(0.0, 0.5_f64 / self.beta()).expect("Distribution not valide, check beta");
        let new_e_field = EField::new_deterministe(&self.lattice(), rng, &d);
        if self.lattice().get_number_of_points() != new_e_field.len() {
            panic!("Length of EField not compatible")
        }
        self.set_e_field(new_e_field.project_to_gauss(self.link_matrix(), self.lattice()).unwrap());
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField<D>;
    
    /// Replace the electrical field with the given input. It should panic if the input is not of the correct size.
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_points()`
    fn set_e_field(&mut self, e_field: EField<D>);
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize;
    
    /// get the derivative \partial_t U(link)
    fn get_derivative_u(link: &LatticeLinkCanonical<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3>;
    /// get the derivative \partial_t E(point)
    fn get_derivative_e(point: &LatticePoint<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>) -> Option<VectorN<Su3Adjoint, D>>;
    
    /// Get the energy of the conjugate momenta configuration
    fn get_hamiltonian_efield(&self) -> Real;
    
    /// Get the total energy, by default [`LatticeHamiltonianSimulationState::get_hamiltonian_efield`]
    /// + [`LatticeState::get_hamiltonian_links`]
    fn get_hamiltonian_total(&self) -> Real {
        self.get_hamiltonian_links() + self.get_hamiltonian_efield()
    }
    
    
}

/// Trait to create a simulation state
pub trait LatticeHamiltonianSimulationStateNew<D>
    where Self: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// Create a new simulation state
    fn new(lattice: LatticeCyclique<D>, beta: Real, e_field: EField<D>, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError>;
    
    /// Ceate a new state with e_field randomly distributed as [`rand_distr::Normal`]
    /// # Panic
    /// Panics if N(0, 0.5/beta ) is not a valide distribution (for exampple beta = 0)
    fn new_random_e(lattice: LatticeCyclique<D>, beta: Real, link_matrix: LinkMatrix, rng: &mut impl rand::Rng) -> Result<Self, SimulationError>
    {
        // TODO verify
        // rand_distr::StandardNormal
        let d = rand_distr::Normal::new(0.0, 0.5_f64 / beta).expect("Distribution not valide, check Beta.");
        let e_field = EField::new_deterministe(&lattice, rng, &d).project_to_gauss(&link_matrix, &lattice).unwrap();
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
}

/// [`LatticeHamiltonianSimulationState`] who represent link matrices at the same time position as its conjugate momenta
/// `e_field`.
///
/// If you have a LatticeState and want the default way of adding the conjugate momenta and doing simulation look at
/// [`LatticeHamiltonianSimulationStateSyncDefault`].
///
/// I would adivce of implementing this trait and not [`SimulationStateLeapFrog`], as there is
/// a wrapper ([`SimulationStateLeap`]) for [`SimulationStateLeapFrog`].
/// Also not implementing both trait gives you a compile time verification that you did not
/// considered a leap frog state as a sync one.
pub trait SimulationStateSynchrone<D>
    where Self: LatticeHamiltonianSimulationState<D> + Clone,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// does half a step for the conjugate momenta.
    fn simulate_to_leapfrog<I, State>(&self, delta_t: Real, integrator: &I) -> Result<State, SimulationError>
        where State: SimulationStateLeapFrog<D>,
        I: SymplecticIntegrator<Self, State, D>
    {
        integrator.integrate_sync_leap(&self, delta_t)
    }
    
    /// does `number_of_steps` with `delta_t` at each step using a leap_frog algorithm by fist
    /// doing half a setp and then finishing by doing half setp
    fn simulate_using_leapfrog_n<I, State>(
        &self,
        delta_t: Real,
        number_of_steps: usize,
        integrator: &I,
    ) -> Result<Self, SimulationError>
        where State: SimulationStateLeapFrog<D>,
        I: SymplecticIntegrator<Self, State, D>,
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
    
    /// Does the same thing as [`SimulationStateSynchrone::simulate_using_leapfrog_n`]
    /// but use the default wrapper [`SimulationStateLeap`] for the leap frog state
    fn simulate_using_leapfrog_n_auto<I>(
        &self,
        delta_t: Real,
        number_of_steps: usize,
        integrator: &I,
    ) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<Self, SimulationStateLeap<Self, D>, D>,
    {
        self.simulate_using_leapfrog_n(delta_t, number_of_steps, integrator)
    }
    
    /// Does a simulation step using the sync algorithm
    fn simulate_sync<I, T>(&self, delta_t: Real, integrator: &I) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<Self, T, D>,
        T: SimulationStateLeapFrog<D>,
    {
        integrator.integrate_sync_sync(&self, delta_t)
    }
    
    /// Does `numbers_of_times` of step of size `delta_t` using the sync algorithm
    fn simulate_sync_n<I, T>(&self, delta_t: Real, integrator: &I, numbers_of_times: usize) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<Self, T, D>,
        T: SimulationStateLeapFrog<D>,
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

/// [`LatticeHamiltonianSimulationState`] who represent link matrices at time T and its conjugate momenta at time T + 1/2
///
/// If you have a [`SimulationStateSynchrone`] look at the wrapper [`SimulationStateLeap`].
pub trait SimulationStateLeapFrog<D>
    where Self: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// Simulate the state to synchrone by finishing the half setp
    fn simulate_to_synchrone<I, State>(&self, delta_t: Real, integrator: &I) -> Result<State, SimulationError>
        where State: SimulationStateSynchrone<D>,
        I: SymplecticIntegrator<State, Self, D>
    {
        integrator.integrate_leap_sync(&self, delta_t)
    }
    
    /// Does one simulation step using the leap frog algorithm
    fn simulate_leap<I, T>(&self, delta_t: Real, integrator: &I) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<T, Self, D>,
        T: SimulationStateSynchrone<D>,
    {
        integrator.integrate_leap_leap(&self, delta_t)
    }
    
    /// does `numbers_of_times` simulation set of size `delta_t` using the leap frog algorithm.
    fn simulate_leap_n<I, T>(&self, delta_t: Real, integrator: &I, numbers_of_times: usize) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<T, Self, D>,
        T: SimulationStateSynchrone<D>,
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
///
/// It has the default pure gauge hamiltonian
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeStateDefault<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    lattice : LatticeCyclique<D>,
    beta: Real,
    link_matrix: LinkMatrix,
}

impl<D> LatticeStateDefault<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// Create a cold configuration. i.e. all the links are set to the unit matrix.
    ///
    /// With the lattice of size `size` and dimension `number_of_points` ( see [`LatticeCyclique::new`] )
    /// and beta parameter `beta`.
    pub fn new_cold(size: Real, beta: Real , number_of_points: usize) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        Self::new(lattice, beta, link_matrix)
    }
    
    /// Create a "hot" configuration, i.e. the link matrices are chosen randomly.
    ///
    /// With the lattice of size `size` and dimension `number_of_points` ( see [`LatticeCyclique::new`] )
    /// and beta parameter `beta`.
    ///
    /// The creation is determiste in the sence that it is reproducible:
    /// # Example
    /// This example demonstrate how to reproduce the same configuration
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::{simulation::LatticeStateDefault, lattice::LatticeCyclique, dim};
    /// use rand::{SeedableRng, rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// assert_eq!(
    ///     LatticeStateDefault::<dim::U4>::new_deterministe(1_f64, 1_f64, 4, &mut rng_1).unwrap(),
    ///     LatticeStateDefault::<dim::U4>::new_deterministe(1_f64, 1_f64, 4, &mut rng_2).unwrap()
    /// );
    /// ```
    pub fn new_deterministe(
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points)
            .ok_or(SimulationError::InitialisationError)?;
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng);
        Self::new(lattice, beta, link_matrix)
    }
    
    /// Correct the numerical drift, reprojecting all the link matrices to SU(3).
    /// see [`LinkMatrix::normalize`].
    pub fn normalize_link_matrices(&mut self) {
        self.link_matrix.normalize()
    }
    
    /// Get a mutable reference to the link matrix at `link`
    pub fn get_link_mut(&mut self, link: &LatticeLinkCanonical<D>) -> Option<&mut CMatrix3> {
        let index = link.to_index(&self.lattice);
        if index < self.link_matrix.len(){
            Some(&mut self.link_matrix[index])
        }
        else{
            None
        }
    }
}

impl<D> LatticeStateNew<D> for LatticeStateDefault<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn new(lattice: LatticeCyclique<D>, beta: Real, link_matrix: LinkMatrix) -> Result<Self, SimulationError> {
        if lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice, link_matrix, beta})
    }
}

impl<D> LatticeState<D> for LatticeStateDefault<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    const CA: Real = 3_f64;
    
    getter_trait!(
        /// The link matrices of this state.
        link_matrix, LinkMatrix
    );
    getter_trait!(lattice, LatticeCyclique<D>);
    getter_copy_trait!(beta, Real);
    
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_canonical_links_space()`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        if self.lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            panic!("Link matrices are not of the correct size");
        }
        self.link_matrix = link_matrix;
    }
    
    /// Get the default pure gauge Hamiltonian.
    /// # Panic
    /// Panic if plaquettes cannot be found
    fn get_hamiltonian_links(&self) -> Real {
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::get_all_positive_directions().iter().map(|dir_i| {
                Direction::get_all_positive_directions().iter()
                    .filter(|dir_j| dir_i.to_index() < dir_j.to_index())
                    .map(|dir_j| {
                        1_f64 - self.link_matrix().get_pij(&el, dir_i, dir_j, self.lattice())
                            .expect("Plaquette not found").trace().real() / Self::CA
                    }).sum::<Real>()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
}

/// Depreciated use [`LatticeHamiltonianSimulationStateSyncDefault`] using [`LatticeStateDefault::<U4>`] instead.
#[derive(Debug, PartialEq, Clone)]
#[deprecated(
    since = "0.1.0",
    note = "Please use `LatticeHamiltonianSimulationStateSyncDefault<LatticeStateDefault<dim::U4>>` instead"
)]
pub struct LatticeHamiltonianSimulationStateSync {
    lattice : LatticeCyclique<na::U4>,
    beta: Real,
    e_field: EField<na::U4>,
    link_matrix: LinkMatrix,
    t: usize,
}

#[allow(deprecated)]
impl SimulationStateSynchrone<na::U4> for LatticeHamiltonianSimulationStateSync {}

#[allow(deprecated)]
impl LatticeHamiltonianSimulationStateSync {
    
    /// Generate a hot (i.e. random) initial state.
    ///
    /// Single threaded generation with a given random number generator.
    /// `size` is the size parameter of the lattice and `number_of_points` is the number of points
    /// in each spatial dimension of the lattice. See [`LatticeCyclique::new`] for more info.
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
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
    
    /// Generate a configuration with cold e_field and hot link matrices
    pub fn new_deterministe_cold_e_hot_link (
        size: Real,
        beta: Real,
        number_of_points: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points)
            .ok_or(SimulationError::InitialisationError)?;
        let e_field = EField::new_cold(&lattice);
        let link_matrix = LinkMatrix::new_deterministe(&lattice, rng);
        
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
    
    /// Generate a hot (i.e. random) initial state.
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
            let link_matrix = LinkMatrix::new_random_threaded(&lattice, number_of_thread - 1)
                .map_err(SimulationError::ThreadingError)?;
            
            let e_field = handel.join().map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
            // TODO not very clean: imporve
            Ok(Self::new(lattice, beta, e_field, link_matrix, 0)?)
        }).map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
        return result;
    }
    
    /// Generate a new cold state.
    ///
    /// It meas that the link matrices are set to the identity and electrical field are set to 0
    pub fn new_cold(size: Real, beta: Real , number_of_points: usize) -> Result<Self, SimulationError> {
        let lattice = LatticeCyclique::new(size, number_of_points).ok_or(SimulationError::InitialisationError)?;
        let link_matrix = LinkMatrix::new_cold(&lattice);
        let e_field = EField::new_cold(&lattice);
        Self::new(lattice, beta, e_field, link_matrix, 0)
    }
    
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn get_gauss(&self, point: &LatticePoint<na::U4>) -> Option<CMatrix3> {
        self.e_field().get_gauss(self.link_matrix(), point, self.lattice())
    }
}

#[allow(deprecated)]
impl LatticeState<na::U4> for LatticeHamiltonianSimulationStateSync {
    
    const CA: Real = 3_f64;
    
    getter_trait!(
        /// The link matrices of this state.
        link_matrix, LinkMatrix
    );
    getter_trait!(lattice, LatticeCyclique<na::U4>);
    getter_copy_trait!(beta, Real);
    
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_canonical_links_space()`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        if self.lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            panic!("Link matrices are not of the correct size");
        }
        self.link_matrix = link_matrix
    }
    
    /// Get the Hamiltonian of the state.
    /// # Panic
    /// Panic if plaquettes cannot be found
    fn get_hamiltonian_links(&self) -> Real {
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::get_all_positive_directions().iter().map(|dir_i| {
                Direction::get_all_positive_directions().iter()
                    .filter(|dir_j| dir_i.to_index() < dir_j.to_index())
                    .map(|dir_j| {
                        1_f64 - self.link_matrix().get_pij(&el, dir_i, dir_j, self.lattice())
                            .expect("Plaquette not found").trace().real() / Self::CA
                    }).sum::<Real>()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
}

#[allow(deprecated)]
impl LatticeHamiltonianSimulationStateNew<na::U4> for LatticeHamiltonianSimulationStateSync{
    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    fn new(lattice: LatticeCyclique<na::U4>, beta: Real, e_field: EField<na::U4>, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        if lattice.get_number_of_points() != e_field.len() ||
            lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice, e_field, link_matrix, t, beta})
    }
}

#[allow(deprecated)]
impl LatticeHamiltonianSimulationState<na::U4> for LatticeHamiltonianSimulationStateSync {
    
    /// # Panic
    /// Panic if EField cannot be found
    fn get_hamiltonian_efield(&self) -> Real {
        // TODO optimize
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::get_all_positive_directions().iter().map(|dir_i| {
                let e_i = self.e_field().get_e_field(&el, dir_i, self.lattice()).expect("EField not found").to_matrix();
                (e_i * e_i).trace().real()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField<na::U4> {
        &self.e_field
    }
    
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_points()`
    fn set_e_field(&mut self, e_field: EField<na::U4>) {
        if self.lattice.get_number_of_points() != e_field.len() {
            panic!("e_field is not of the correct size")
        }
        self.e_field = e_field;
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.t
    }
    
    /// Get the derive of U_i(x).
    fn get_derivative_u(link: &LatticeLinkCanonical<na::U4>, link_matrix: &LinkMatrix, e_field: &EField<na::U4>, lattice: &LatticeCyclique<na::U4>) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * Self::CA ).sqrt();
        let u_i = link_matrix.get_matrix(&LatticeLink::from(*link), lattice)?;
        let e_i = e_field.get_e_field(link.pos(), link.dir(), lattice)?;
        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / lattice.size()));
    }
    
    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    fn get_derivative_e(point: &LatticePoint<na::U4>, link_matrix: &LinkMatrix, _e_field: &EField<na::U4>, lattice: &LatticeCyclique<na::U4>) -> Option<Vector4<Su3Adjoint>> {
        let c = - (2_f64 / Self::CA).sqrt();
        let dir_pos = Direction::get_all_positive_directions();
        let iterator = dir_pos.iter().map(|dir| {
            let u_i = link_matrix.get_matrix(&LatticeLink::new(*point, *dir), lattice)?;
            let sum_s: CMatrix3 = Direction::get_all_directions().iter()
                .filter(|dir_2| dir_2.to_positive() != *dir)
                .map(|dir_2| {
                    link_matrix.get_sij(point, dir, dir_2, lattice)
                        .map(|el| el.adjoint())
                }).sum::<Option<CMatrix3>>()?;
            Some(Su3Adjoint::new(
                Vector8::<Real>::from_fn(|index, _| {
                    c * (su3::GENERATORS[index] * u_i * sum_s).trace().imaginary() / lattice.size()
                })
            ))
        });
        let mut return_vector = Vector4::from_element(Su3Adjoint::default());
        for (index, element) in iterator.enumerate() {
            return_vector[index] = element?;
        }
        Some(return_vector)
    }
    
}

/// wrapper for a simulation state using leap frog ([`SimulationStateLeap`]) using a synchrone type ([`SimulationStateSynchrone`]).
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SimulationStateLeap<State, D>
    where State: SimulationStateSynchrone<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    state: State,
    _phantom: PhantomData<D>,
}

impl<State, D> SimulationStateLeap<State, D>
    where State: SimulationStateSynchrone<D> + LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    getter!(state, State);
    
    /// Create a leap state from a sync one by integrating by half a setp the e_field
    pub fn from_synchrone<I>(s: &State, integrator: &I , delta_t: Real) -> Result<Self, SimulationError>
        where I: SymplecticIntegrator<State, Self, D>
    {
        s.simulate_to_leapfrog(delta_t, integrator)
    }
    
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn get_gauss(&self, point: &LatticePoint<D>) -> Option<CMatrix3> {
        self.e_field().get_gauss(self.link_matrix(), point, self.lattice())
    }
}

/// This state is a leap frog state
impl<State, D> SimulationStateLeapFrog<D> for SimulationStateLeap<State, D>
    where State: SimulationStateSynchrone<D> + LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{}

    /// We just transmit the function of `State`, there is nothing new.
impl<State, D> LatticeState<D> for SimulationStateLeap<State, D>
    where State: LatticeHamiltonianSimulationState<D> + SimulationStateSynchrone<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    const CA: Real = State::CA;
    
    /// The link matrices of this state.
    fn link_matrix(&self) -> &LinkMatrix {
        self.state().link_matrix()
    }
    /// # Panic
    /// panic under the same condition as `State::set_link_matrix`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        self.state.set_link_matrix(link_matrix)
    }
    
    fn lattice(&self) -> &LatticeCyclique<D> {
        self.state().lattice()
    }
    
    fn beta(&self) -> Real {
        self.state().beta()
    }
    
    fn get_hamiltonian_links(&self) -> Real {
        self.state().get_hamiltonian_links()
    }
}

impl<State, D> LatticeHamiltonianSimulationStateNew<D> for SimulationStateLeap<State, D>
    where State: LatticeHamiltonianSimulationState<D> + SimulationStateSynchrone<D> + LatticeHamiltonianSimulationStateNew<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    fn new(lattice: LatticeCyclique<D>, beta: Real, e_field: EField<D>, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        let state = State::new(lattice, beta, e_field, link_matrix, t)?;
        Ok(Self {state, _phantom: PhantomData})
    }
}

/// We just transmit the function of `State`, there is nothing new.
impl<State, D> LatticeHamiltonianSimulationState<D> for SimulationStateLeap<State, D>
    where State: LatticeHamiltonianSimulationState<D> + SimulationStateSynchrone<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    
    project!(get_hamiltonian_efield, state, Real);
    
    project!(
        /// The "Electrical" field of this state.
        e_field, state, &EField<D>
    );
    
    project_mut!(
        /// # Panic
        /// panic under the same condition as `State::set_e_field`
        set_e_field, state, (), e_field: EField<D>
    );
    
    project!(
        /// return the time state, i.e. the number of time the simulation ran.
        t, state, usize
    );
    
    fn get_derivative_u(link: &LatticeLinkCanonical<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3> {
        State::get_derivative_u(link, link_matrix, e_field, lattice)
    }
    
    fn get_derivative_e(point: &LatticePoint<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>) -> Option<VectorN<Su3Adjoint, D>> {
        State::get_derivative_e(point, link_matrix, e_field, lattice)
    }
    
}

/// wrapper to implement [`LatticeHamiltonianSimulationState`] from a [`LatticeState`] using
/// the default implementation of conjugate momenta.
///
/// It also implement [`SimulationStateSynchrone`].
#[derive(Debug, PartialEq, Clone)]
pub struct LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    lattice_state: State,
    e_field: EField<D>,
    t: usize,
}

#[cfg(feature = "serde-serialize")]
impl<State, D> serde::Serialize for LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D>,
    D: na::DimName,
    na::DefaultAllocator: na::base::allocator::Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Sync + Send,
    na::DefaultAllocator: na::base::allocator::Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    EField<D>: Serialize,
    State: Serialize + Clone,
    Direction<D>: DirectionList,
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
        where T: serde::Serializer,
    {
        (self.lattice_state.clone(), self.e_field.clone(), self.t).serialize(serializer)
    }
}

#[cfg(feature = "serde-serialize")]
impl<'de, State, D> serde::Deserialize<'de> for LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D>,
    D: na::DimName,
    na::DefaultAllocator: na::base::allocator::Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Sync + Send,
    na::DefaultAllocator: na::base::allocator::Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    EField<D>: Deserialize<'de>,
    State: Deserialize<'de>,
    Direction<D>: DirectionList,
{
    fn deserialize<T>(deserializer: T) -> Result<Self, T::Error>
        where T: serde::Deserializer<'de>,
    {
        serde::Deserialize::deserialize(deserializer).map(|(lattice_state, e_field, t)| {
            Self {lattice_state, e_field, t}
        })
    }
}


impl<State, D> LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    pub fn get_state_owned(self) -> State {
        self.lattice_state
    }
    
    pub fn lattice_state(&self) -> &State {
        &self.lattice_state
    }
    
    /// # Panic
    /// Panics if N(0, 0.5/beta ) is not a valide distribution (for exampple beta = 0)
    pub fn new_random_e_state(lattice_state: State, rng: &mut impl rand::Rng) -> Self {
        let d = rand_distr::Normal::new(0.0, 0.5_f64 / lattice_state.beta()).expect("Distribution not valide, check Beta.");
        let e_field = EField::new_deterministe(&lattice_state.lattice(), rng, &d)
            .project_to_gauss(lattice_state.link_matrix(), lattice_state.lattice()).unwrap();
        Self{lattice_state, e_field, t: 0}
    }
    
    pub fn new_e_cold(lattice_state: State) -> Self {
        let e_field = EField::new_cold(&lattice_state.lattice());
        Self{lattice_state, e_field, t: 0}
    }
}

impl<State, D> LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where Self: LatticeHamiltonianSimulationState<D>,
    State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn get_gauss(&self, point: &LatticePoint<D>) -> Option<CMatrix3> {
        self.e_field.get_gauss(self.link_matrix(), point, self.lattice())
    }
}

/// This is an sync State
impl<State, D> SimulationStateSynchrone<D> for LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D> + Clone,
    Self: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{}

impl<State, D> LatticeState<D> for LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    
    fn link_matrix(&self) -> &LinkMatrix{
        self.lattice_state.link_matrix()
    }
    
    /// # Panic
    /// panic under the same condition as `State::set_link_matrix`
    fn set_link_matrix(&mut self, link_matrix: LinkMatrix) {
        self.lattice_state.set_link_matrix(link_matrix)
    }
    
    fn lattice(&self) -> &LatticeCyclique<D> {
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

impl<State, D> LatticeHamiltonianSimulationStateNew<D> for LatticeHamiltonianSimulationStateSyncDefault<State, D>
    where State: LatticeState<D> + LatticeStateNew<D>,
    Self: LatticeHamiltonianSimulationState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    fn new(lattice: LatticeCyclique<D>, beta: Real, e_field: EField<D>, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        if lattice.get_number_of_points() != e_field.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice_state: State::new(lattice, beta, link_matrix)?, e_field, t})
    }
}

impl<D> LatticeHamiltonianSimulationState<D> for LatticeHamiltonianSimulationStateSyncDefault<LatticeStateDefault<D>, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    /// By default \sum_x Tr(E_i E_i)
    fn get_hamiltonian_efield(&self) -> Real {
        // TODO optimize
        self.lattice().get_points().par_bridge().map(|el| {
            Direction::get_all_positive_directions().iter().map(|dir_i| {
                let e_i = self.e_field().get_e_field(&el, dir_i, self.lattice()).unwrap().to_matrix();
                (e_i * e_i).trace().real()
            }).sum::<Real>()
        }).sum::<Real>() * self.beta()
    }
    
    /// The "Electrical" field of this state.
    fn e_field(&self) -> &EField<D> {
        &self.e_field
    }
    
    /// # Panic
    /// Panic if the length of link_matrix is different from `lattice.get_number_of_points()`
    fn set_e_field(&mut self, e_field: EField<D>) {
        if self.lattice().get_number_of_points() != e_field.len() {
            panic!("e_field is not of the correct size")
        }
        self.e_field = e_field;
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    fn t(&self) -> usize {
        self.t
    }
    
    /// Get the derive of U_i(x).
    fn get_derivative_u(link: &LatticeLinkCanonical<D>, link_matrix: &LinkMatrix, e_field: &EField<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * Self::CA ).sqrt();
        let u_i = link_matrix.get_matrix(&LatticeLink::from(*link), lattice)?;
        let e_i = e_field.get_e_field(link.pos(), link.dir(), lattice)?;
        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / lattice.size()));
    }
    
    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    fn get_derivative_e(point: &LatticePoint<D>, link_matrix: &LinkMatrix, _e_field: &EField<D>, lattice: &LatticeCyclique<D>) -> Option<VectorN<Su3Adjoint, D>> {
        let c = - (2_f64 / Self::CA).sqrt();
        let dir_pos = Direction::<D>::get_all_positive_directions();
        let iterator = dir_pos.iter().map(|dir| {
            let u_i = link_matrix.get_matrix(&LatticeLink::new(*point, *dir), lattice)?;
            let sum_s: CMatrix3 = Direction::<D>::get_all_directions().iter()
                .filter(|dir_2| dir_2.to_positive() != *dir)
                .map(|dir_2| {
                    link_matrix.get_sij(point, dir, dir_2, lattice)
                        .map(|el| el.adjoint())
                }).sum::<Option<CMatrix3>>()?;
            Some(Su3Adjoint::new(
                Vector8::<Real>::from_fn(|index, _| {
                    c * (su3::GENERATORS[index] * u_i * sum_s).trace().imaginary() / lattice.size()
                })
            ))
        });
        let mut return_vector = VectorN::<_, D>::from_element(Su3Adjoint::default());
        for (index, element) in iterator.enumerate() {
            return_vector[index] = element?;
        }
        Some(return_vector)
    }
    
}
