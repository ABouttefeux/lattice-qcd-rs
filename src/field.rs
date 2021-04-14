
//! Represent the fields on the lattice.
//!

use super::{
    Real,
    Complex,
    CMatrix3,
    lattice::{
        LatticePoint,
        LatticeCyclique,
        LatticeLink,
        Direction,
        LatticeElementToIndex,
        DirectionList,
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
    utils::levi_civita,
};
use na::{
    Matrix3,
    ComplexField,
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
};
use std::{
    ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign, Div, DivAssign, Sub, SubAssign, Neg},
    vec::Vec,
};
use rayon::prelude::*;
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};


/// Adjoint representation of SU(3), it is su(3) (i.e. the lie algebra).
/// See [`su3::GENERATORS`] to view the order of generators.
/// Note that the generators are normalize such that `Tr[T^a T^b] = \delta^{ab} / 2`
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Su3Adjoint {
    data: Vector8<Real>
}

#[allow(clippy::len_without_is_empty)]
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
            .map(|(pos, el)| *GENERATORS[pos] * na::Complex::<Real>::from(el))
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
        // NOTE: should it consume ? the user can manually clone and there is use because
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
    
    /// Returns the trace squared `Tr(X^2)`.
    ///
    /// It is more accurate and faster than computing
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # use lattice_qcd_rs::ComplexField;
    /// # let m = Su3Adjoint::default();
    /// (m.to_matrix() * m.to_matrix()).trace().real();
    /// ```
    #[inline]
    pub fn trace_squared(&self) -> Real {
        // TODO investigate.
        self.data.iter().map(|el| el * el).sum::<Real>() / 2_f64
    }
    
    /// Return the t coeff `t = - 1/2 * Tr(X^2)`.
    /// If you are looking for the trace square use [Self::trace_squared] instead.
    ///
    /// It is used for [`su3::su3_exp_i`].
    /// # Example
    /// ```
    /// # extern crate nalgebra;
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::from([1_f64; 8]);
    /// let m = su3.to_matrix();
    /// assert_eq!(nalgebra::Complex::new(su3.t(), 0_f64), - nalgebra::Complex::from(0.5_f64) * (m * m).trace());
    /// ```
    #[inline]
    pub fn t(&self) -> Real {
        - 0.5_f64 * self.trace_squared()
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
    #[inline]
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
    
    /// Get an iterator over the ellements.
    pub fn iter(&self) -> impl Iterator<Item = &Real> {
        self.data.iter()
    }
    
    /// Get an iterator over the mutable ref of ellements.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Real> {
        self.data.iter_mut()
    }
    
    /// Get a mutlable reference over the data.
    pub fn data_mut(&mut self) -> &mut Vector8<Real> {
        &mut self.data
    }
}

impl<'a> IntoIterator for &'a Su3Adjoint {
    type Item = &'a Real;
    type IntoIter = <&'a Vector8<Real> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Su3Adjoint {
    type Item = &'a mut Real;
    type IntoIter = <&'a mut Vector8<Real> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl AddAssign for Su3Adjoint {
    fn add_assign(&mut self, other: Self) {
        self.data += other.data();
    }
}

impl Add<Su3Adjoint> for Su3Adjoint {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output{
        self += rhs;
        self
    }
}

impl Add<&Su3Adjoint> for Su3Adjoint {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output{
        self + *rhs
    }
}

impl Add<Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn add(self, rhs: Su3Adjoint) -> Self::Output{
        rhs + self
    }
}

impl Add<&Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn add(self, rhs: &Su3Adjoint) -> Self::Output{
        self + *rhs
    }
}

impl MulAssign<f64> for Su3Adjoint {
    fn mul_assign(&mut self, rhs: f64) {
        self.data *= rhs;
    }
}

impl Mul<Real> for Su3Adjoint {
    type Output = Self;
    fn mul(mut self, rhs: Real) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<&Real> for Su3Adjoint {
    type Output = Self;
    fn mul(self, rhs: &Real) -> Self::Output {
        self * (*rhs)
    }
}

impl Mul<Real> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn mul(self, rhs: Real) -> Self::Output {
        *self * rhs
    }
}

impl Mul<&Real> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn mul(self, rhs: &Real) -> Self::Output {
        *self * rhs
    }
}

impl Mul<Su3Adjoint> for Real {
    type Output = Su3Adjoint;
    fn mul(self, rhs: Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Mul<&Su3Adjoint> for Real {
    type Output = Su3Adjoint;
    fn mul(self, rhs: &Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Mul<Su3Adjoint> for &Real {
    type Output = Su3Adjoint;
    fn mul(self, rhs: Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Mul<&Su3Adjoint> for &Real {
    type Output = Su3Adjoint;
    fn mul(self, rhs: &Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl DivAssign<f64> for Su3Adjoint {
    fn div_assign(&mut self, rhs: f64) {
        self.data /= rhs;
    }
}

impl DivAssign<&f64> for Su3Adjoint {
    fn div_assign(&mut self, rhs: &f64) {
        self.data /= *rhs;
    }
}

impl Div<Real> for Su3Adjoint {
    type Output = Self;
    fn div(mut self, rhs: Real) -> Self::Output {
        self /= rhs;
        self
    }
}

impl Div<&Real> for Su3Adjoint {
    type Output = Self;
    fn div(self, rhs: &Real) -> Self::Output {
        self / (*rhs)
    }
}

impl Div<Real> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn div(self, rhs: Real) -> Self::Output {
        *self / rhs
    }
}

impl Div<&Real> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn div(self, rhs: &Real) -> Self::Output {
        *self / rhs
    }
}

impl SubAssign for Su3Adjoint {
    fn sub_assign(&mut self, other: Self) {
        self.data -= other.data();
    }
}

impl Sub<Su3Adjoint> for Su3Adjoint {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output{
        self -= rhs;
        self
    }
}

impl Sub<&Su3Adjoint> for Su3Adjoint {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self::Output{
        self - *rhs
    }
}

impl Sub<Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn sub(self, rhs: Su3Adjoint) -> Self::Output{
        rhs - self
    }
}

impl Sub<&Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn sub(self, rhs: &Su3Adjoint) -> Self::Output{
        *self - rhs
    }
}

impl Neg for Su3Adjoint {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Su3Adjoint::new(- self.data)
    }
}

impl Neg for &Su3Adjoint {
    type Output = Su3Adjoint;
    fn neg(self) -> Self::Output {
        Su3Adjoint::new(- self.data())
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

impl From<&Su3Adjoint> for Vector8<Real> {
    fn from(v: &Su3Adjoint) -> Self {
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
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
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
    /// # use lattice_qcd_rs::dim::U4;
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let lattice = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
    /// assert_eq!(
    ///     LinkMatrix::new_deterministe(&lattice, &mut rng_1),
    ///     LinkMatrix::new_deterministe(&lattice, &mut rng_2)
    /// );
    /// ```
    pub fn new_deterministe<D>(
        l: &LatticeCyclique<D>,
        rng: &mut impl rand::Rng,
    ) -> Self
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        // l.get_links_space().map(|_| Su3Adjoint::random(rng, d).to_su3()).collect()
        // using a for loop imporves performance. ( probably because the vector is pre allocated).
        let mut data = Vec::with_capacity(l.get_number_of_canonical_links_space());
        for _ in l.get_links() {
            // the iterator *should* be in order
            let matrix = su3::get_random_su3(rng);
            data.push(matrix);
        }
        Self {data}
    }
    
    
    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reduced easily, In that case use [`LinkMatrix::new_random_threaded`].
    ///
    /// # Errors
    /// Returns [`ThreadError::ThreadNumberIncorect`] if `number_of_thread` is 0.
    pub fn new_random_threaded<D>(
        l: &LatticeCyclique<D>,
        number_of_thread: usize,
    ) -> Result<Self, ThreadError>
        where D: DimName + Eq,
        DefaultAllocator: Allocator<usize, D> + Allocator<na::Complex<f64>, na::U3, na::U3>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        if number_of_thread == 0 {
            return Err(ThreadError::ThreadNumberIncorect);
        }
        else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return Ok(LinkMatrix::new_deterministe(l, &mut rng));
        }
        let data = run_pool_parallel_vec_with_initialisation_mutable(
            l.get_links(),
            &(),
            &|rng, _, _| su3::get_random_su3(rng),
            rand::thread_rng,
            number_of_thread,
            l.get_number_of_canonical_links_space(),
            l,
            &CMatrix3::zeros(),
        )?;
        Ok(Self {data})
    }
    
    /// Create a cold configuration ( where the link matrices is set to 1).
    pub fn new_cold<D>(l: &LatticeCyclique<D>) -> Self
        where D: DimName,
        DefaultAllocator: Allocator<usize, D> + Allocator<na::Complex<f64>, na::U3, na::U3>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        Self {data: vec![CMatrix3::identity(); l.get_number_of_canonical_links_space()]}
    }
    
    /// get the link matrix associated to given link using the notation
    /// $`U_{-i}(x) = U^\dagger_{i}(x-i)`$
    pub fn get_matrix<D>(&self, link: &LatticeLink<D>, l: &LatticeCyclique<D>) -> Option<Matrix3<na::Complex<Real>>>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let link_c = l.get_canonical(*link);
        let matrix = self.data.get(link_c.to_index(l))?;
        if link.is_dir_negative() {
            // that means the the link was in the negative direction
            Some(matrix.adjoint())
        }
        else {
            Some(*matrix)
        }
    }
    
    /// Get $`S_ij(x) = U_j(x) U_i(x+j) U^\dagger_j(x+i)`$.
    pub fn get_sij<D>(&self, point: &LatticePoint<D>, dir_i: &Direction<D>, dir_j: &Direction<D>, lattice: &LatticeCyclique<D>) -> Option<Matrix3<na::Complex<Real>>>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let u_j = self.get_matrix(&LatticeLink::new(*point, *dir_j), lattice)?;
        let point_pj = lattice.add_point_direction(*point, dir_j);
        let u_i_p_j = self.get_matrix(&LatticeLink::new(point_pj, *dir_i), lattice)?;
        let point_pi = lattice.add_point_direction(*point, dir_i);
        let u_j_pi_d = self.get_matrix(&LatticeLink::new(point_pi, *dir_j), lattice)?.adjoint();
        Some(u_j * u_i_p_j * u_j_pi_d)
    }
    
    /// Get the plaquette $`P_{ij}(x) = U_i(x) S^\dagger_ij(x)`$.
    pub fn get_pij<D>(&self, point: &LatticePoint<D>, dir_i: &Direction<D>, dir_j: &Direction<D>, lattice: &LatticeCyclique<D>) -> Option<Matrix3<na::Complex<Real>>>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let s_ij = self.get_sij(point, dir_i, dir_j, lattice)?;
        let u_i = self.get_matrix(&LatticeLink::new(*point, *dir_i), lattice)?;
        Some(u_i * s_ij.adjoint())
    }
    
    /// Take the average of the trace of all plaquettes
    #[allow(clippy::as_conversions)] // no try into for f64
    pub fn average_trace_plaquette<D>(&self, lattice: &LatticeCyclique<D>) -> Option<na::Complex<Real>>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        if lattice.get_number_of_canonical_links_space() != self.len() {
            return None;
        }
        // the order does not matter as we sum
        let sum = lattice.get_points().par_bridge().map(|point| {
            Direction::get_all_positive_directions().iter().map( |dir_i|{
                Direction::get_all_positive_directions().iter().filter(|dir_j| dir_i.to_index() < dir_j.to_index())
                    .map(|dir_j|{
                        self.get_pij(&point, dir_i, dir_j, lattice).map(|el| el.trace())
                    }).sum::<Option<na::Complex<Real>>>()
            }).sum::<Option<na::Complex<Real>>>()
        }).sum::<Option<na::Complex<Real>>>()?;
        let number_of_directions = (D::dim() * (D::dim() - 1)) / 2;
        let number_of_plaquette = (lattice.get_number_of_points() * number_of_directions) as f64;
        Some(sum / number_of_plaquette)
    }
    
    /// Get the clover, used for F_mu_nu tensor
    pub fn get_clover<D>(&self, point: &LatticePoint<D>, dir_i: &Direction<D>, dir_j: &Direction<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3>
    where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        Some(self.get_pij(point, dir_i, dir_j, lattice)? + self.get_pij(point, dir_j, &-dir_i, lattice)? + self.get_pij(point, &-dir_i, &-dir_j, lattice)? + self.get_pij(point, &-dir_j, dir_i, lattice)?)
    }
    
    /// Get the `F^{ij}` tensor using the clover appropriation. The direction are set to positive
    /// See arXive:1512.02374.
    pub fn get_f_mu_nu<D>(&self, point: &LatticePoint<D>, dir_i: &Direction<D>, dir_j: &Direction<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D> + Allocator<Complex, na::U3, na::U3>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let m = self.get_clover(point, dir_i, dir_j, lattice)? - self.get_clover(point, dir_j, dir_i, lattice)?;
        Some(m / Complex::from(8_f64 * lattice.size() * lattice.size()))
    }
    
    /// Get the chromomagentic field at a given point
    pub fn get_magnetic_field_vec<D>(&self, point: &LatticePoint<D>, lattice: &LatticeCyclique<D>) -> Option<VectorN<CMatrix3, D>>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D> + Allocator<Complex, na::U3, na::U3> + Allocator<CMatrix3, D>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let mut vec = VectorN::<CMatrix3, D>::zeros();
        for dir in Direction::<D>::get_all_positive_directions() {
            vec[dir.to_index()] = self.get_magnetic_field(point, dir, lattice)?;
        }
        Some(vec)
    }
    
    /// Get the chromomagentic field at a given point alongisde a given direction
    pub fn get_magnetic_field<D>(&self, point: &LatticePoint<D>, dir: &Direction<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D> + Allocator<Complex, na::U3, na::U3>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let sum = Direction::<D>::get_all_positive_directions().iter().map(|dir_i| {
            Direction::<D>::get_all_positive_directions().iter().map(|dir_j| {
                let f_mn = self.get_f_mu_nu(point, dir_i, dir_j, lattice)?;
                let lc = Complex::from(levi_civita(&[dir.to_index(), dir_i.to_index(), dir_j.to_index()]).to_f64());
                Some(f_mn * lc)
            }).sum::<Option<CMatrix3>>()
        }).sum::<Option<CMatrix3>>()?;
        Some(sum / Complex::new(0_f64, 2_f64))
    }
    
    /// Get the chromomagentic field at a given point alongisde a given direction given by lattice link
    pub fn get_magnetic_field_link<D>(&self, link: &LatticeLink<D>, lattice: &LatticeCyclique<D>) -> Option<Matrix3<na::Complex<Real>>>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D> + Allocator<Complex, na::U3, na::U3>,
        VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        self.get_magnetic_field(link.pos(), link.dir(), lattice)
    }
    
    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Returns wether the there is no data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Correct the numerical drift, reprojecting all the matrices to SU(3).
    pub fn normalize(&mut self) {
        self.data.par_iter_mut().for_each(|el| {
            su3::orthonormalize_matrix_mut(el);
        });
    }
    
    /// Iter on the data
    pub fn iter(&self) -> impl Iterator<Item = &CMatrix3> {
        self.data.iter()
    }
    
    /// Iter mutably on the data
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut CMatrix3> {
        self.data.iter_mut()
    }
}

impl<'a> IntoIterator for &'a LinkMatrix {
    type Item = &'a CMatrix3;
    type IntoIter = <&'a Vec<CMatrix3> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut LinkMatrix {
    type Item = &'a mut CMatrix3;
    type IntoIter = <&'a mut Vec<CMatrix3> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl Index<usize> for LinkMatrix {
    type Output = CMatrix3;
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl IndexMut<usize> for LinkMatrix {
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output{
        &mut self.data[pos]
    }
}

/// Represent an electric field.
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct EField<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
{
    #[cfg_attr(feature = "serde-serialize", serde(bound(serialize = "VectorN<Su3Adjoint, D>: Serialize", deserialize = "VectorN<Su3Adjoint, D>: Deserialize<'de>")) )]
    data: Vec<VectorN<Su3Adjoint, D>>, // use a Vec<[Su3Adjoint; 4]> instead ?
}

impl<D> EField<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    
    /// Create a new "Electrical" field.
    pub fn new (data: Vec<VectorN<Su3Adjoint, D>>) -> Self {
        Self {data}
    }
    
    /// Get the raw data.
    pub fn data(&self) -> &Vec<VectorN<Su3Adjoint, D>> {
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
    /// # use lattice_qcd_rs::dim::U4;
    /// use rand::{SeedableRng,rngs::StdRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(- 1_f64..1_f64);
    /// let lattice = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
    /// assert_eq!(
    ///     EField::new_deterministe(&lattice, &mut rng_1, &distribution),
    ///     EField::new_deterministe(&lattice, &mut rng_2, &distribution)
    /// );
    /// ```
    pub fn new_deterministe(
        l: &LatticeCyclique<D>,
        rng: &mut impl rand::Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut data = Vec::with_capacity(l.get_number_of_points());
        for _ in l.get_points() {
            // iterator *should* be ordoned
            data.push(VectorN::<Su3Adjoint, D>::from_fn(|_, _| Su3Adjoint::random(rng, d)));
        }
        Self {data}
    }
    
    /// Single thread generation by seeding a new rng number.
    /// To create a seedable and reproducible set use [`EField::new_deterministe`]..
    pub fn new_random(
        l: &LatticeCyclique<D>,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        EField::new_deterministe(l, &mut rng, d)
    }
    
    /// Create a new cold configuration for the electriccal field, i.e. all E ar set to 0.
    pub fn new_cold(l: &LatticeCyclique<D>) -> Self {
        let p1 = Su3Adjoint::new_from_array([0_f64; 8]);
        Self {data: vec![VectorN::<Su3Adjoint, D>::from_element(p1); l.get_number_of_points()]}
    }
    
    /// Get `E(point) = [E_x(point), E_y(point), E_z(point)]`.
    pub fn get_e_vec(&self, point: &LatticePoint<D>, l: &LatticeCyclique<D>) -> Option<&VectorN<Su3Adjoint, D>> {
        self.data.get(point.to_index(l))
    }
    
    /// Get `E_{dir}(point)`. The sign of the direction does not change the output. i.e.
    /// `E_{-dir}(point) = E_{dir}(point)`.
    pub fn get_e_field(&self, point: &LatticePoint<D>, dir: &Direction<D>, l: &LatticeCyclique<D>) -> Option<&Su3Adjoint> {
        let value = self.get_e_vec(point, l);
        match value {
            Some(vec) => vec.get(dir.to_index()),
            None => None,
        }
    }
    
    /// Returns wether there is no data
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Return the Gauss parameter `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    #[inline]
    pub fn get_gauss(&self, link_matrix: &LinkMatrix, point: &LatticePoint<D>, lattice: &LatticeCyclique<D>) -> Option<CMatrix3> {
        if lattice.get_number_of_points() != self.len() || lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return None;
        }
        Direction::get_all_positive_directions().iter().map(|dir| {
            let e_i = self.get_e_field(point, dir, lattice)?;
            let u_mi = link_matrix.get_matrix(&LatticeLink::new(*point, - *dir), lattice)?;
            let p_mi = lattice.add_point_direction(*point, &- dir);
            let e_m_i = self.get_e_field(&p_mi, dir, lattice)?;
            Some(e_i.to_matrix() - u_mi * e_m_i.to_matrix() * u_mi.adjoint())
        }).sum::<Option<CMatrix3>>()
    }
    
    /// Get the deviation from the Gauss law
    #[inline]
    pub fn get_gauss_sum_div(&self, link_matrix: &LinkMatrix, lattice: &LatticeCyclique<D>) -> Option<Real> {
        if lattice.get_number_of_points() != self.len() || lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return None;
        }
        lattice.get_points().par_bridge().map(|point| {
            self.get_gauss(link_matrix, &point, lattice).map(|el| (su3::GENERATORS.iter().copied().sum::<CMatrix3>() * el).trace().abs())
        }).sum::<Option<Real>>()
    }
    
    /// project to that the gauss law is approximatively respected ( up to `f64::EPSILON * 10` per point)
    #[allow(clippy::as_conversions)] // no try into for f64
    #[inline]
    pub fn project_to_gauss(&self, link_matrix: &LinkMatrix, lattice: &LatticeCyclique<D>) -> Option<Self> {
        // TODO improve
        
        const NUMBER_FOR_LOOP: usize = 4;
        
        if lattice.get_number_of_points() != self.len() || lattice.get_number_of_canonical_links_space() != link_matrix.len() {
            return None;
        }
        let mut return_val = self.project_to_gauss_step(link_matrix, lattice);
        loop {
            let val_dif = return_val.get_gauss_sum_div(link_matrix, lattice)?;
            //println!("diff : {}", val_dif);
            if val_dif.is_nan() {
                return None;
            }
            if val_dif <= f64::EPSILON * (lattice.get_number_of_points() * 4 * 8 * 10) as f64 {
                break;
            }
            for _ in 0_usize..NUMBER_FOR_LOOP {
                return_val = return_val.project_to_gauss_step(link_matrix, lattice);
                //println!("{}", return_val[0][0][0]);
            }
        }
        Some(return_val)
    }
    
    /// Done one step to project to gauss law
    /// # Panic
    /// panics if the link matric and lattice is not of the correct size.
    #[inline]
    fn project_to_gauss_step(&self, link_matrix: &LinkMatrix, lattice: &LatticeCyclique<D>) -> Self {
        /// see https://arxiv.org/pdf/1512.02374.pdf
        // TODO verify
        const K: na::Complex<f64> = na::Complex::new(0.12_f64, 0_f64);
        let data = lattice.get_points().collect::<Vec<LatticePoint<D>>>().par_iter().map(|point| {
            let e = self.get_e_vec(&point, lattice).unwrap();
            VectorN::<_, D>::from_fn(|index_dir, _| {
                let dir = Direction::<D>::get_all_positive_directions()[index_dir];
                let u = link_matrix.get_matrix(&LatticeLink::new(*point, dir), lattice).unwrap();
                let gauss = self.get_gauss(link_matrix, &point, lattice).unwrap();
                let gauss_p = self.get_gauss(link_matrix, &lattice.add_point_direction(*point, &dir), lattice).unwrap();
                Su3Adjoint::new(Vector8::from_fn( |index, _| {
                    2_f64 * ( su3::GENERATORS[index] * (( u * gauss * u.adjoint() * gauss_p - gauss) * K + su3::GENERATORS[index] * na::Complex::from(e[dir.to_index()][index]) )).trace().real()
                }))
            })
        }).collect();
        Self::new(data)
    }
}

impl<'a, D> IntoIterator for &'a EField<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    type Item = &'a VectorN<Su3Adjoint, D>;
    type IntoIter = <&'a Vec<VectorN<Su3Adjoint, D>> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, D> IntoIterator for &'a mut EField<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
    Direction<D>: DirectionList,
{
    type Item = &'a mut VectorN<Su3Adjoint, D>;
    type IntoIter = <&'a mut Vec<VectorN<Su3Adjoint, D>> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<D> Index<usize> for EField<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
{
    type Output = VectorN<Su3Adjoint, D>;
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl<D> IndexMut<usize> for EField<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
    DefaultAllocator: Allocator<Su3Adjoint, D>,
    VectorN<Su3Adjoint, D>: Sync + Send,
{
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output{
        &mut self.data[pos]
    }
}

#[cfg(test)]
mod test{
    use super::*;
    use rand::SeedableRng;
    use approx::*;
    use super::super::{
        Complex,
        dim::U3,
        lattice::*,
    };
    
    
    const EPSILON: f64 = 0.000_000_001_f64;
    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;
    
    #[test]
    fn test_get_e_field_pos_neg() {
        use super::super::lattice;
        
        let l = LatticeCyclique::new(1_f64, 4).unwrap();
        let e = EField::new(vec![VectorN::<_, na::U4>::from([Su3Adjoint::from([1_f64; 8]), Su3Adjoint::from([2_f64; 8]), Su3Adjoint::from([3_f64; 8]), Su3Adjoint::from([2_f64; 8]) ]) ]);
        assert_eq!(
            e.get_e_field(&LatticePoint::new([0, 0, 0, 0].into()), &lattice::DirectionEnum::XPos.into(), &l),
            e.get_e_field(&LatticePoint::new([0, 0, 0, 0].into()), &lattice::DirectionEnum::XNeg.into(), &l)
        );
    }
    
    #[test]
    fn test_su3_adj(){
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let d = rand::distributions::Uniform::from(-1_f64..1_f64);
        for _ in 0..100 {
            let v = Su3Adjoint::random(&mut rng, &d);
            let m = v.to_matrix();
            assert_abs_diff_eq!(v.trace_squared(), (m * m).trace().modulus(), epsilon=EPSILON);
            assert_eq_complex!(v.d(), nalgebra::Complex::new(0_f64, 1_f64) * m.determinant(), EPSILON);
            assert_eq_complex!(v.t(), -(m * m).trace() / Complex::from(2_f64), EPSILON);
        }
    }
    
    #[test]
    fn mangetic_field() {
        let lattice = LatticeCyclique::<U3>::new(1_f64, 4).unwrap();
        let mut link_matrix = LinkMatrix::new_cold(&lattice);
        let point = LatticePoint::from([0, 0, 0]);
        let dir_x = Direction::new(0, true).unwrap();
        let dir_y = Direction::new(1, true).unwrap();
        let dir_z = Direction::new(2, true).unwrap();
        let clover = link_matrix.get_clover(&point, &dir_x, &dir_y, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::from(4_f64), clover, EPSILON);
        let f = link_matrix.get_f_mu_nu(&point, &dir_x, &dir_y, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), f, EPSILON);
        let b = link_matrix.get_magnetic_field(&point, &dir_x, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        // ---
        link_matrix[0] = CMatrix3::identity() * Complex::new(0_f64, 1_f64);
        let clover = link_matrix.get_clover(&point, &dir_x, &dir_y, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new(2_f64, 0_f64), clover, EPSILON);
        let clover = link_matrix.get_clover(&point, &dir_y, &dir_x, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new(2_f64, 0_f64), clover, EPSILON);
        let f = link_matrix.get_f_mu_nu(&point, &dir_x, &dir_y, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new(0_f64, 0_f64), f, EPSILON);
        let b = link_matrix.get_magnetic_field(&point, &dir_x, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        //--
        let mut link_matrix = LinkMatrix::new_cold(&lattice);
        let link = LatticeLinkCanonical::new([1, 0 , 0].into(), dir_y).unwrap();
        link_matrix[link.to_index(&lattice)] = CMatrix3::identity() * Complex::new(0_f64, 1_f64);
        let clover = link_matrix.get_clover(&point, &dir_x, &dir_y, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new(3_f64, 1_f64), clover, EPSILON);
        let clover = link_matrix.get_clover(&point, &dir_y, &dir_x, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new(3_f64, -1_f64), clover, EPSILON);
        let f = link_matrix.get_f_mu_nu(&point, &dir_x, &dir_y, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new(0_f64, 0.25_f64), f, EPSILON);
        let b = link_matrix.get_magnetic_field(&point, &dir_x, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        let b = link_matrix.get_magnetic_field(&point, &dir_z, &lattice).unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::new( 0.25_f64, 0_f64), b, EPSILON);
        let b_2 = link_matrix.get_magnetic_field(&[4, 0, 0].into(), &dir_z, &lattice).unwrap();
        assert_eq_matrix!(b, b_2, EPSILON);
    }
}
