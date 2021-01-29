
//! Represent the fields on the lattice.
//!

use super::{
    Real,
    CMatrix3,
    lattice::{
        LatticeLinkCanonical,
        LatticePoint,
        LatticeCyclique,
        LatticeLink,
        Direction,
        LatticeElementToIndex,
    },
    Vector8,
    su3,
    su3::{
        GENERATORS,
    },
    I,
    ZERO,
    Complex,
    thread::{
        ThreadError,
        run_pool_parallel_vec_with_initialisation_mutable,
    },
    integrator::Integrator,
};
use na::{
    Vector3,
    Matrix3,
    ComplexField
};
use  std::{
    ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign},
    vec::Vec,
};
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use crossbeam::thread;



/// Adjoint representation of SU(3), it is su(3) (i.e. the lie algebra).
/// See [`su3::GENERATORS`] to view the order of generators.
/// Note that the generators are normalize such that `Tr[T^a T^b] = \delta^{ab} / 2`
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Su3Adjoint {
    data: Vector8<Real>
}

impl Su3Adjoint {
    
    /// create a new Su3Adjoint representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// extern crate nalgebra;
    /// use lattice_qcd_rs::field::Su3Adjoint;
    ///
    /// let su3 = Su3Adjoint::new(nalgebra::VectorN::<f64, nalgebra::U8>::from_element(1_f64));
    /// ```
    pub const fn new(data: Vector8<Real>) -> Self {
        Self {data}
    }
    
    /// create a new Su3Adjoint representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// ```
    pub fn new_from_array(data: [Real; 8]) -> Self {
        Su3Adjoint::new(Vector8::from(data))
    }
    
    /// get the data inside the Su3Adjoint.
    pub const fn data(&self) -> &Vector8<Real> {
        &self.data
    }
    
    /// return the su(3) (Lie algebra) matrix.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::Su3Adjoint};
    /// let su3 = Su3Adjoint::new_from_array([1_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64]);
    /// assert_eq!(su3.to_matrix(), *lattice_qcd_rs::su3::GENERATORS[0]);
    /// ```
    pub fn to_matrix(&self) -> Matrix3<na::Complex<Real>> {
        self.data.iter().enumerate()
            .map(|(pos, el)|  *GENERATORS[pos] * na::Complex::<Real>::from(el))
            .sum()
    }
    
    /// Return the SU(3) matrix associated with this generator.
    /// Note that the function consume self.
    /// # Example
    /// ```
    /// # extern crate nalgebra;
    /// # use lattice_qcd_rs::{field::Su3Adjoint};
    /// let su3 = Su3Adjoint::new_from_array([1_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64]);
    /// assert_eq!(su3.to_su3().determinant(), nalgebra::Complex::from(1_f64));
    /// ```
    pub fn to_su3(self) -> Matrix3<na::Complex<Real>> {
        // TODO should it consume ? the user can manually clone and there is use because
        // where the value is not necessary anymore.
        su3::su3_exp_i(self)
    }
    
    /// return exp( T^a v^a) where v is self.
    /// Note that the function consume self.
    pub fn exp(self) -> Matrix3<na::Complex<Real>> {
        su3::su3_exp_r(self)
    }
    
    /// create a new random SU3 adjoint.
    /// # Example
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    ///
    /// let mut rng = rand::thread_rng();
    /// let distribution = rand::distributions::Uniform::from(- 1_f64..1_f64);
    /// let su3 = Su3Adjoint::random(&mut rng, &distribution);
    /// ```
    pub fn random(rng: &mut impl rand::Rng, d: &impl rand_distr::Distribution<Real>) -> Self {
        Self {
            data : Vector8::<Real>::from_fn(|_,_| d.sample(rng))
        }
    }
    
    /// Return the t coeff `t = 1/2 * Tr(X^2)`.
    /// Used for [`su3::su3_exp_i`]
    /// # Example
    /// ```
    /// # extern crate nalgebra;
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::from([1_f64; 8]);
    /// let m = su3.to_matrix();
    /// assert_eq!(su3.t(), - nalgebra::Complex::from(0.5_f64) * (m * m).trace());
    /// ```
    pub fn t(&self) -> na::Complex<Real> {
        // todo optimize
        let m = self.to_matrix();
        - na::Complex::from(0.5_f64) * (m * m).trace()
    }
    
    /// Return the t coeff `d = i * det(X)`.
    /// Used for [`su3::su3_exp_i`]
    /// # Example
    /// ```
    /// # extern crate nalgebra;
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::from([1_f64; 8]);
    /// let m = su3.to_matrix();
    /// assert_eq!(su3.d(), nalgebra::Complex::new(0_f64, 1_f64) * m.determinant());
    /// ```
    pub fn d(&self) -> na::Complex<Real> {
        self.to_matrix().determinant() * I
    }
    
    /// Return the number of data. This number is 8
    /// ```
    /// # extern crate nalgebra;
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let su3 = Su3Adjoint::new(nalgebra::VectorN::<f64, nalgebra::U8>::zeros());
    /// assert_eq!(su3.len(), 8);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Mul<Real> for Su3Adjoint {
    type Output = Self;
    fn mul(mut self, rhs: Real) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<Su3Adjoint> for Real {
    type Output = Su3Adjoint;
    fn mul(self, rhs: Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Add<Su3Adjoint> for Su3Adjoint {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output{
        self += rhs;
        self
    }
}

impl AddAssign for Su3Adjoint {
    fn add_assign(&mut self, other: Self) {
        self.data += other.data()
    }
}

impl MulAssign<f64> for Su3Adjoint {
    fn mul_assign(&mut self, rhs: f64) {
        self.data *= rhs;
    }
}


/// Return the representation for the zero matrix.
impl Default for Su3Adjoint {
    /// Return the representation for the zero matrix.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// assert_eq!(Su3Adjoint::default(), Su3Adjoint::new_from_array([0_f64; 8]));
    /// ```
    fn default() -> Self{
        Su3Adjoint::new(Vector8::from_element(0_f64))
    }
}

impl Index<usize> for Su3Adjoint {
    type Output = Real;
    
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound (greater or equal to 8).
    /// ```should_panic
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// let _ = su3[8];
    /// ```
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl IndexMut<usize> for Su3Adjoint {
    
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound (greater or equal to 8).
    /// ```should_panic
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let mut su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// su3[8] += 1_f64;
    /// ```
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output{
        &mut self.data[pos]
    }
}

impl From<Vector8<Real>> for Su3Adjoint {
    fn from(v: Vector8<Real>) -> Self {
        Su3Adjoint::new(v)
    }
}

impl From<Su3Adjoint> for Vector8<Real> {
    fn from(v: Su3Adjoint) -> Self {
        v.data
    }
}

impl From<[Real; 8]> for Su3Adjoint {
    fn from(v: [Real; 8]) -> Self {
        Su3Adjoint::new_from_array(v)
    }
}

/// Represents the link matrices
#[derive(Debug, PartialEq, Clone)]
pub struct LinkMatrix {
    data: Vec<Matrix3<na::Complex<Real>>>,
}

impl LinkMatrix {
    
    /// Creat a new link matrix field
    pub const fn new (data: Vec<Matrix3<na::Complex<Real>>>) -> Self{
        Self {data}
    }
    
    /// Get the raw data.
    pub const fn data(&self) -> &Vec<Matrix3<na::Complex<Real>>> {
        &self.data
    }
    
    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data but slower than [`LinkMatrix::new_random_threaded`].
    /// # Example
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclique};
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(- 1_f64..1_f64);
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// assert_eq!(
    ///     LinkMatrix::new_deterministe(&lattice, &mut rng_1, &distribution),
    ///     LinkMatrix::new_deterministe(&lattice, &mut rng_2, &distribution)
    /// );
    /// ```
    pub fn new_deterministe(
        l: &LatticeCyclique,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        // l.get_links_space().map(|_|  Su3Adjoint::random(rng, d).to_su3()).collect()
        // using a for loop imporves performance. ( probably because the vector is pre allocated).
        let mut data = Vec::with_capacity(l.get_number_of_canonical_links_space());
        for _ in l.get_links_space() {
            // the iterator *should* be in order
            let matrix = Su3Adjoint::random(rng, d).to_su3();
            data.push(matrix);
        }
        Self {data}
    }
    
    
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reduced easily, In that case use [`LinkMatrix::new_random_threaded`].
    pub fn new_random_threaded<Distribution>(
        l: &LatticeCyclique,
        d: &Distribution,
        number_of_thread: usize,
    ) -> Result<Self, ThreadError>
        where Distribution: rand_distr::Distribution<Real> + Sync,
    {
        if number_of_thread == 0 {
            return Err(ThreadError::ThreadNumberIncorect);
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return Ok(LinkMatrix::new_deterministe(l, &mut rng, d));
        }
        let data = run_pool_parallel_vec_with_initialisation_mutable(
            l.get_links_space(),
            d,
            &|rng, _, d| Su3Adjoint::random(rng, d).to_su3(),
            || rand::thread_rng(),
            number_of_thread,
            l.get_number_of_canonical_links_space(),
            l,
            CMatrix3::zeros(),
        )?;
        Ok(Self {data})
    }
    
    /// get the link matrix associated to given link using the notation
    /// $`U_{-i}(x) = U^\dagger_{i}(x-i)`$
    pub fn get_matrix(&self, link: &LatticeLink, l: &LatticeCyclique)-> Option<Matrix3<na::Complex<Real>>> {
        let link_c = l.get_canonical(link);
        let matrix = self.data.get(link_c.to_index(l))? ;
        if link.is_dir_negative() {
            // that means the the link was in the negative direction
            return Some(matrix.adjoint());
        }
        else {
            return Some(*matrix);
        }
    }
    
    /// Get $`S_ij(x) = U_j(x) U_i(x+j) U^\dagger_j(x+i)`$.
    pub fn get_sij(&self, point: &LatticePoint, dir_i: &Direction, dir_j: &Direction, lattice: &LatticeCyclique) -> Option<Matrix3<na::Complex<Real>>> {
        let u_j = self.get_matrix(&LatticeLink::new(point.clone(), dir_j.clone()), lattice)?;
        let point_pj = lattice.add_point_direction(point.clone(), dir_j);
        let u_i_p_j = self.get_matrix(&LatticeLink::new(point_pj, dir_i.clone()), lattice)?;
        let point_pi = lattice.add_point_direction(point.clone(), dir_i);
        let u_j_pi_d = self.get_matrix(&LatticeLink::new(point_pi, dir_j.clone()), lattice)?.adjoint();
        Some(u_j * u_i_p_j * u_j_pi_d)
    }
    
    /// Get the plaquette $`P_{ij}(x) = U_i(x) S^\dagger_ij(x)`$.
    pub fn get_pij(&self, point: &LatticePoint, dir_i: &Direction, dir_j: &Direction, lattice: &LatticeCyclique) -> Option<Matrix3<na::Complex<Real>>> {
        let s_ij = self.get_sij(point, dir_i, dir_j, lattice)?;
        let u_i = self.get_matrix(&LatticeLink::new(point.clone(), dir_i.clone()), lattice)?;
        Some(u_i * s_ij.adjoint())
    }
    
    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
}

/// Represent an electric field.
#[derive(Debug, PartialEq, Clone)]
pub struct EField
{
    data: Vec<Vector3<Su3Adjoint>>, // use a Vec<[Su3Adjoint; 4]> instead ?
}

impl EField {
    
    /// Create a new "Electrical" field.
    pub const fn new (data: Vec<Vector3<Su3Adjoint>>) -> Self {
        Self {data}
    }
    
    /// Get the raw data.
    pub const fn data(&self) -> &Vec<Vector3<Su3Adjoint>> {
        &self.data
    }
    
    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data.
    /// # Example
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::{field::EField, lattice::LatticeCyclique};
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(- 1_f64..1_f64);
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// assert_eq!(
    ///     EField::new_deterministe(&lattice, &mut rng_1, &distribution),
    ///     EField::new_deterministe(&lattice, &mut rng_2, &distribution)
    /// );
    /// ```
    pub fn new_deterministe(
        l: &LatticeCyclique,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut data = Vec::with_capacity(l.get_number_of_points());
        for _ in l.get_points() {
            // iterator *should* be ordoned
            let p1 = Su3Adjoint::random(rng, d);
            let p2 = Su3Adjoint::random(rng, d);
            let p3 = Su3Adjoint::random(rng, d);
            data.push(Vector3::new(p1, p2, p3));
        }
        Self {data}
    }
    
    /// Single thread generation by seeding a new rng number. Used in [`LatticeSimulationState::new_random_threaded`].
    /// To create a seedable and reproducible set use [`EField::new_deterministe`]..
    pub fn new_random(
        l: &LatticeCyclique,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        EField::new_deterministe(l, &mut rng, d)
    }
    
    /// Get `E(point) = [E_x(point), E_y(point), E_z(point)]`.
    pub fn get_e_vec(&self, point: &LatticePoint, l: &LatticeCyclique) -> Option<&Vector3<Su3Adjoint>> {
        self.data.get(point.to_index(l))
    }
    
    /// Get `E_{dir}(point)`. The sign of the direction does not change the output. i.e. `E_{-dir}(point) = E_{dir}(point)`.
    pub fn get_e_field(&self, point: &LatticePoint, dir: &Direction, l: &LatticeCyclique) -> Option<&Su3Adjoint> {
        let value = self.get_e_vec(point, l);
        match value {
            Some(vec) => Some(&vec[dir.to_index()]),
            None => None,
        }
    }
}

#[cfg(test)]
#[test]
fn test_get_e_field_pos_neg() {
    let l = LatticeCyclique::new(1_f64, 4).unwrap();
    let e = EField::new(vec![Vector3::from([Su3Adjoint::from([1_f64; 8]), Su3Adjoint::from([2_f64; 8]), Su3Adjoint::from([3_f64; 8]) ]) ]);
    assert_eq!(
        e.get_e_field(&LatticePoint::new([0, 0, 0]), &Direction::XPos, &l),
        e.get_e_field(&LatticePoint::new([0, 0, 0]), &Direction::XNeg, &l)
    );
}

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

/// Represent a simulation state at a set time.
#[derive(Debug, PartialEq, Clone)]
pub struct LatticeSimulationState {
    lattice : LatticeCyclique,
    e_field: EField,
    link_matrix: LinkMatrix,
    t: usize,
}

impl LatticeSimulationState {
    
    /// create a new simulation state. If `e_field` or `link_matrix` does not have the corresponding
    /// amount of data compared to lattice it fails to create the state.
    /// `t` is the number of time the simulation ran. i.e. the time sate.
    pub fn new(lattice: LatticeCyclique, e_field: EField, link_matrix: LinkMatrix, t: usize) -> Result<Self, SimulationError> {
        if lattice.get_number_of_points() != e_field.len() ||
            lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return Err(SimulationError::InitialisationError);
        }
        Ok(Self {lattice, e_field, link_matrix, t})
    }
    
    /// Generate a random initial state.
    ///
    /// Single threaded generation with a given random number generator.
    /// `size` is the size parameter of the lattice and `number_of_points` is the number of points
    /// in each spatial dimension of the lattice. See [LatticeCyclique::new] for more info.
    ///
    /// useful to reproduce a set of data but slower than [`LatticeSimulationState::new_random_threaded`]
    /// # Example
    /// ```
    /// extern crate rand;
    /// extern crate rand_distr;
    /// # use lattice_qcd_rs::{field::LatticeSimulationState, lattice::LatticeCyclique};
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(- 1_f64..1_f64);
    /// assert_eq!(
    ///     LatticeSimulationState::new_deterministe(1_f64, 4, &mut rng_1, &distribution).unwrap(),
    ///     LatticeSimulationState::new_deterministe(1_f64, 4, &mut rng_2, &distribution).unwrap()
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
        LatticeSimulationState::new(lattice, e_field, link_matrix, 0)
    }
    
    /// Generate a random initial state.
    ///
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reproduce easily, In that case use [`LatticeSimulationState::new_deterministe`].
    pub fn new_random_threaded<Distribution>(
        size: Real,
        number_of_points: usize,
        d: &Distribution,
        number_of_thread : usize
    ) -> Result<Self, SimulationError>
        where Distribution: rand_distr::Distribution<Real> + Sync,
    {
        if number_of_thread == 0 {
            return Err(SimulationError::ThreadingError(ThreadError::ThreadNumberIncorect)); //
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return LatticeSimulationState::new_deterministe(size, number_of_points, &mut rng, d);
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
            Ok(LatticeSimulationState::new(lattice, e_field, link_matrix, 0)?)
        }).map_err(|err| SimulationError::ThreadingError(ThreadError::Panic(err)))?;
        return result;
    }
    
    /// The "Electrical" field of this state.
    pub const fn e_field(&self) -> &EField {
        &self.e_field
    }
    
    /// The link matrices of this state.
    pub const fn link_matrix(&self) -> &LinkMatrix {
        &self.link_matrix
    }
    
    pub const fn lattice(&self) -> &LatticeCyclique {
        &self.lattice
    }
    
    /// return the time state, i.e. the number of time the simulation ran.
    pub const fn t(&self) -> usize {
        self.t
    }
    
    /// The `C_A` constant.
    const CA: Real = 3_f64;
    
    /// Get the derive of U_i(x).
    pub fn get_derivatives_u(&self, link: &LatticeLinkCanonical) -> Option<CMatrix3> {
        let c = Complex::new(0_f64, 2_f64 * LatticeSimulationState::CA ).sqrt();
        let u_i = self.link_matrix().get_matrix(&LatticeLink::from(link.clone()), self.lattice())?;
        let e_i = self.e_field().get_e_field(link.pos(), link.dir(), self.lattice())?;
        return Some(e_i.to_matrix() * u_i * c * Complex::from(1_f64 / self.lattice().size()));
    }
    
    /// Get the derive of E(x) (as a vector of Su3Adjoint).
    pub fn get_derivative_e(&self, point: &LatticePoint) -> Option<Vector3<Su3Adjoint>> {
        let c = - Complex::new(2_f64 / LatticeSimulationState::CA, 0_f64).sqrt();
        let mut derivative = Vector3::from_element(Su3Adjoint::default());
        for dir in Direction::POSITIVES_SPACE.iter() {
            let derivative_i = &mut derivative[dir.to_index()];
            let u_i = self.link_matrix().get_matrix(&LatticeLink::new(point.clone(), dir.clone()), self.lattice())?;
            let mut sum_s = CMatrix3::zeros();
            for dir_2 in Direction::DIRECTIONS_SPACE.iter() {
                if dir_2.to_positive() != *dir {
                    let s_ij = self.link_matrix().get_sij(point, dir, dir_2, self.lattice())?;
                    sum_s += s_ij.adjoint();
                }
            }
            for b in 0..derivative_i.len(){
                let derivative_i_b = &mut derivative_i[b];
                *derivative_i_b += (c * (su3::GENERATORS[b] * u_i * sum_s).trace().imaginary()).real() / self.lattice().size();
            }
        }
        return Some(derivative);
    }
    
    /// Get the gauss coefficient `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    pub fn get_gauss(&self, point: &LatticePoint) -> Option<CMatrix3> {
        let mut g_return = CMatrix3::zeros();
        // TODO use Iter
        for dir in Direction::POSITIVES_SPACE.iter() {
            let e_i = self.e_field().get_e_field(point, dir, self.lattice())?;
            let u_mi = self.link_matrix().get_matrix(&LatticeLink::new(point.clone(), - dir.clone()), self.lattice())?;
            let p_mi = self.lattice().add_point_direction(point.clone(), & - dir.clone());
            let e_m_i = self.e_field().get_e_field(&p_mi, dir, self.lattice())?;
            g_return += e_i.to_matrix() - u_mi * e_m_i.to_matrix() * u_mi.adjoint();
        }
        Some(g_return)
    }
    
    /// Get the Hamiltonian of the state.
    pub fn get_hamiltonian(&self) -> Real {
        // todo iter$
        // here it is ok to use par_bridge() as we do not care for the order
        self.lattice().get_points().par_bridge().map(|el| {
            let mut sum_plaquette = 0_f64;
            let mut sum_trace_e = 0_f64;
            for dir_i in Direction::POSITIVES_SPACE.iter(){
                for dir_j in Direction::POSITIVES_SPACE.iter(){
                    if dir_i.to_index() < dir_j.to_index() {
                        sum_plaquette += 1_f64 - self.link_matrix().get_pij(&el, dir_i, dir_j, self.lattice()).unwrap().trace().real() / LatticeSimulationState::CA;
                    }
                }
                // TODO optimize
                let e_i = self.e_field().get_e_field(&el, dir_i, self.lattice()).unwrap().to_matrix();
                sum_trace_e += (e_i * e_i).trace().real();
            }
            return sum_plaquette + sum_trace_e;
        }).sum()
    }
    
    /// Do one simulation step.
    pub fn simulate<I>(&self, delta_t: Real, integrator: I) -> Result<Self, SimulationError>
        where I: Integrator
    {
        integrator.integrate(&self, delta_t)
    }
}
