
//! Represent the fields on the lattice.
//!

use super::{
    Real,
    CMatrix3,
    lattice::{
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
    thread::{
        ThreadError,
        run_pool_parallel_vec_with_initialisation_mutable,
    },
};
use na::{
    Vector3,
    Matrix3,
};
use  std::{
    ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign, Div, DivAssign, Sub, SubAssign, Neg},
    vec::Vec,
};
use simba::scalar::RealField;



/// Adjoint representation of SU(3), it is su(3) (i.e. the lie algebra).
/// See [`su3::GENERATORS`] to view the order of generators.
/// Note that the generators are normalize such that `Tr[T^a T^b] = \delta^{ab} / 2`
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Su3Adjoint<F>
    where F: RealField
{
    data: Vector8<F>
}

#[allow(clippy::len_without_is_empty)]
impl<F> Su3Adjoint<F>
    where F: RealField
{
    
    /// create a new Su3Adjoint representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// extern crate nalgebra;
    /// use lattice_qcd_rs::field::Su3Adjoint;
    ///
    /// let su3 = Su3Adjoint::new(nalgebra::VectorN::<f64, nalgebra::U8>::from_element(1_f64));
    /// ```
    pub const fn new(data: Vector8<F>) -> Self {
        Self {data}
    }
    
    /// create a new Su3Adjoint representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// ```
    pub fn new_from_array(data: [F; 8]) -> Self {
        Su3Adjoint::new(Vector8::from(data))
    }
    
    /// get the data inside the Su3Adjoint.
    pub const fn data(&self) -> &Vector8<F> {
        &self.data
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
    pub fn random(rng: &mut impl rand::Rng, d: &impl rand_distr::Distribution<F>) -> Self {
        Self {
            data : Vector8::<F>::from_fn(|_,_| d.sample(rng))
        }
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
    pub fn d(&self) -> na::Complex<F> {
        self.to_matrix().determinant() * na::Complex::i()
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
    
    /// Get an iterator over the ellements.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.data.iter()
    }
    
    /// Get an iterator over the mutable ref of ellements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.data.iter_mut()
    }
    
    /// Get a mutlable reference over the data.
    pub fn data_mut(&mut self) -> &mut Vector8<F> {
        &mut self.data
    }
}

impl<F> Su3Adjoint<F>
    where F: RealField + From<f64>
{
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
    pub fn t(&self) -> na::Complex<F> {
        // todo optimize
        let m = self.to_matrix();
        - na::Complex::from(F::from(0.5_f64)) * (m * m).trace()
    }
    
    /// return the su(3) (Lie algebra) matrix.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::Su3Adjoint};
    /// let su3 = Su3Adjoint::new_from_array([1_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64]);
    /// assert_eq!(su3.to_matrix(), *lattice_qcd_rs::su3::GENERATORS[0]);
    /// ```
    pub fn to_matrix(&self) -> Matrix3<na::Complex<F>> {
        self.data.iter().enumerate()
            .map(|(pos, el)| *GENERATORS[pos] * na::Complex::<F>::from(el))
            .sum()
    }
}

impl<F> Mul<Real> for Su3Adjoint<F>
    where F: RealField
{
    type Output = Self;
    fn mul(mut self, rhs: Real) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<Su3Adjoint<Real>> for Real
{
    type Output = Su3Adjoint<Real>;
    fn mul(self, rhs: Su3Adjoint<Real>) -> Self::Output {
        rhs * self
    }
}

impl<F> Add<Su3Adjoint<F>> for Su3Adjoint<F>
    where F: RealField
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output{
        self += rhs;
        self
    }
}

impl<F> AddAssign for Su3Adjoint<F>
    where F: RealField,
{
    fn add_assign(&mut self, other: Self) {
        self.data += other.data()
    }
}

impl<F> MulAssign<F> for Su3Adjoint<F>
    where F: RealField
{
    fn mul_assign(&mut self, rhs: f64) {
        self.data *= rhs;
    }
}


impl<F> Div<F> for Su3Adjoint<F>
    where F: RealField
{
    type Output = Self;
    fn div(mut self, rhs: Real) -> Self::Output {
        self /= rhs;
        self
    }
}

impl<F> DivAssign<F> for Su3Adjoint<F>
    where F: RealField
{
    fn div_assign(&mut self, rhs: f64) {
        self.data /= rhs;
    }
}

impl<F> Sub<Su3Adjoint<F>> for Su3Adjoint<F>
    where F: RealField
{
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output{
        self -= rhs;
        self
    }
}

impl<F> SubAssign for Su3Adjoint<F>
    where F: RealField
{
    fn sub_assign(&mut self, other: Self) {
        self.data -= other.data()
    }
}

impl<F> Neg for Su3Adjoint<F>
    where F: RealField
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Su3Adjoint::new(- self.data)
    }
}

/// Return the representation for the zero matrix.
impl<F> Default for Su3Adjoint<F>
    where F: RealField + Default
{
    /// Return the representation for the zero matrix.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// assert_eq!(Su3Adjoint::default(), Su3Adjoint::new_from_array([0_f64; 8]));
    /// ```
    fn default() -> Self{
        Su3Adjoint::new(Vector8::from_element(F::default()))
    }
}

impl<F> Index<usize> for Su3Adjoint<F>
    where F: RealField
{
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

impl<F> IndexMut<usize> for Su3Adjoint<F>
    where F: RealField
{
    
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

impl<F> From<Vector8<F>> for Su3Adjoint<F>
    where F: RealField
{
    fn from(v: Vector8<F>) -> Self {
        Su3Adjoint::new(v)
    }
}

impl<F> From<Su3Adjoint<F>> for Vector8<F>
    where F: RealField
{
    fn from(v: Su3Adjoint<F>) -> Self {
        v.data
    }
}

impl<F> From<[F; 8]> for Su3Adjoint<F>
    where F: RealField
{
    fn from(v: [F; 8]) -> Self {
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
            rand::thread_rng,
            number_of_thread,
            l.get_number_of_canonical_links_space(),
            l,
            CMatrix3::zeros(),
        )?;
        Ok(Self {data})
    }
    
    pub fn new_cold(l: &LatticeCyclique) -> Self{
        Self {data: vec![CMatrix3::identity(); l.get_number_of_canonical_links_space()]}
    }
    
    /// get the link matrix associated to given link using the notation
    /// $`U_{-i}(x) = U^\dagger_{i}(x-i)`$
    pub fn get_matrix(&self, link: &LatticeLink, l: &LatticeCyclique)-> Option<Matrix3<na::Complex<Real>>> {
        let link_c = l.get_canonical(*link);
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
        let u_j = self.get_matrix(&LatticeLink::new(*point, *dir_j), lattice)?;
        let point_pj = lattice.add_point_direction(*point, dir_j);
        let u_i_p_j = self.get_matrix(&LatticeLink::new(point_pj, *dir_i), lattice)?;
        let point_pi = lattice.add_point_direction(*point, dir_i);
        let u_j_pi_d = self.get_matrix(&LatticeLink::new(point_pi, *dir_j), lattice)?.adjoint();
        Some(u_j * u_i_p_j * u_j_pi_d)
    }
    
    /// Get the plaquette $`P_{ij}(x) = U_i(x) S^\dagger_ij(x)`$.
    pub fn get_pij(&self, point: &LatticePoint, dir_i: &Direction, dir_j: &Direction, lattice: &LatticeCyclique) -> Option<Matrix3<na::Complex<Real>>> {
        let s_ij = self.get_sij(point, dir_i, dir_j, lattice)?;
        let u_i = self.get_matrix(&LatticeLink::new(*point, *dir_i), lattice)?;
        Some(u_i * s_ij.adjoint())
    }
    
    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Represent an electric field.
#[derive(Debug, PartialEq, Clone)]
pub struct EField
{
    data: Vec<Vector3<Su3Adjoint<Real>>>, // use a Vec<[Su3Adjoint; 4]> instead ?
}

impl EField {
    
    /// Create a new "Electrical" field.
    pub const fn new (data: Vec<Vector3<Su3Adjoint<Real>>>) -> Self {
        Self {data}
    }
    
    /// Get the raw data.
    pub const fn data(&self) -> &Vec<Vector3<Su3Adjoint<Real>>> {
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
    
    
    pub fn new_cold(l: &LatticeCyclique) -> Self {
        let p1 = Su3Adjoint::new_from_array([0_f64; 8]);
        Self {data: vec![Vector3::new(p1, p1, p1); l.get_number_of_points()]}
    }
    /// Get `E(point) = [E_x(point), E_y(point), E_z(point)]`.
    pub fn get_e_vec(&self, point: &LatticePoint, l: &LatticeCyclique) -> Option<&Vector3<Su3Adjoint<Real>>> {
        self.data.get(point.to_index(l))
    }
    
    /// Get `E_{dir}(point)`. The sign of the direction does not change the output. i.e. `E_{-dir}(point) = E_{dir}(point)`.
    pub fn get_e_field(&self, point: &LatticePoint, dir: &Direction, l: &LatticeCyclique) -> Option<&Su3Adjoint<Real>> {
        let value = self.get_e_vec(point, l);
        match value {
            Some(vec) => Some(&vec[dir.to_index()]),
            None => None,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
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
