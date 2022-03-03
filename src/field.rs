//! Represent the fields on the lattice.

use std::iter::{FromIterator, FusedIterator};
use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    vec::Vec,
};

use na::{ComplexField, Matrix3, SVector};
use rayon::iter::FromParallelIterator;
use rayon::prelude::*;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    lattice::{Direction, LatticeCyclique, LatticeElementToIndex, LatticeLink, LatticePoint},
    su3,
    su3::GENERATORS,
    thread::{run_pool_parallel_vec_with_initialisation_mutable, ThreadError},
    utils::levi_civita,
    CMatrix3, Complex, Real, Vector8, I,
};

/// Adjoint representation of SU(3), it is su(3) (i.e. the lie algebra).
/// See [`su3::GENERATORS`] to view the order of generators.
/// Note that the generators are normalize such that `Tr[T^a T^b] = \delta^{ab} / 2`
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Su3Adjoint {
    data: Vector8<Real>,
}

#[allow(clippy::len_without_is_empty)]
impl Su3Adjoint {
    /// create a new Su3Adjoint representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// extern crate nalgebra;
    /// use lattice_qcd_rs::field::Su3Adjoint;
    ///
    /// let su3 = Su3Adjoint::new(nalgebra::SVector::<f64, 8>::from_element(1_f64));
    /// ```
    pub const fn new(data: Vector8<Real>) -> Self {
        Self { data }
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

    /// get the su3 adjoint as a [`Vector8`]
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// #
    /// # let adj = Su3Adjoint::default();
    /// let max = adj.as_vector().max();
    /// let norm = adj.as_ref().norm();
    /// ```
    pub const fn as_vector(&self) -> &Vector8<Real> {
        self.data()
    }

    /// get the su3 adjoint as mut ref to a [`Vector8`]
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// #
    /// # let mut adj = Su3Adjoint::default();
    /// adj.as_vector_mut().apply(|el| el + 1_f64);
    /// adj.as_mut().set_magnitude(1_f64);
    /// ```
    pub fn as_vector_mut(&mut self) -> &mut Vector8<Real> {
        self.data_mut()
    }

    /// return the su(3) (Lie algebra) matrix.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::Su3Adjoint};
    /// let su3 = Su3Adjoint::new_from_array([1_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64]);
    /// assert_eq!(su3.to_matrix(), *lattice_qcd_rs::su3::GENERATORS[0]);
    /// ```
    pub fn to_matrix(self) -> Matrix3<na::Complex<Real>> {
        self.data
            .iter()
            .enumerate()
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
    /// use lattice_qcd_rs::field::Su3Adjoint;
    ///
    /// let mut rng = rand::thread_rng();
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// let su3 = Su3Adjoint::random(&mut rng, &distribution);
    /// ```
    pub fn random<Rng>(rng: &mut Rng, d: &impl rand_distr::Distribution<Real>) -> Self
    where
        Rng: rand::Rng + ?Sized,
    {
        Self {
            data: Vector8::<Real>::from_fn(|_, _| d.sample(rng)),
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
    /// assert_eq!(
    ///     nalgebra::Complex::new(su3.t(), 0_f64),
    ///     -nalgebra::Complex::from(0.5_f64) * (m * m).trace()
    /// );
    /// ```
    #[inline]
    pub fn t(&self) -> Real {
        -0.5_f64 * self.trace_squared()
    }

    /// Return the t coeff `d = i * det(X)`.
    /// Used for [`su3::su3_exp_i`]
    /// # Example
    /// ```
    /// # extern crate nalgebra;
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::from([1_f64; 8]);
    /// let m = su3.to_matrix();
    /// assert_eq!(
    ///     su3.d(),
    ///     nalgebra::Complex::new(0_f64, 1_f64) * m.determinant()
    /// );
    /// ```
    #[inline]
    pub fn d(&self) -> na::Complex<Real> {
        self.to_matrix().determinant() * I
    }

    /// Return the number of data. This number is 8
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let su3 = Su3Adjoint::new(nalgebra::SVector::<f64, 8>::zeros());
    /// assert_eq!(su3.len(), 8);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get an iterator over the ellements.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let adj = Su3Adjoint::default();
    /// let sum_abs = adj.iter().map(|el| el.abs()).sum::<f64>();
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &Real> + ExactSizeIterator + FusedIterator {
        self.data.iter()
    }

    /// Get an iterator over the mutable ref of ellements.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let mut adj = Su3Adjoint::default();
    /// adj.iter_mut().for_each(|el| *el = *el / 2_f64);
    /// ```
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Real> + ExactSizeIterator + FusedIterator {
        self.data.iter_mut()
    }

    /// Get a mutlable reference over the data.
    pub fn data_mut(&mut self) -> &mut Vector8<Real> {
        &mut self.data
    }
}

impl AsRef<Vector8<f64>> for Su3Adjoint {
    fn as_ref(&self) -> &Vector8<f64> {
        self.as_vector()
    }
}

impl AsMut<Vector8<f64>> for Su3Adjoint {
    fn as_mut(&mut self) -> &mut Vector8<f64> {
        self.as_vector_mut()
    }
}

impl<'a> IntoIterator for &'a Su3Adjoint {
    type IntoIter = <&'a Vector8<Real> as IntoIterator>::IntoIter;
    type Item = &'a Real;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Su3Adjoint {
    type IntoIter = <&'a mut Vector8<Real> as IntoIterator>::IntoIter;
    type Item = &'a mut Real;

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

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Su3Adjoint> for Su3Adjoint {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl Add<Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    fn add(self, rhs: Su3Adjoint) -> Self::Output {
        rhs + self
    }
}

impl Add<&Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    fn add(self, rhs: &Su3Adjoint) -> Self::Output {
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

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<&Su3Adjoint> for Su3Adjoint {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl Sub<Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    fn sub(self, rhs: Su3Adjoint) -> Self::Output {
        rhs - self
    }
}

impl Sub<&Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    fn sub(self, rhs: &Su3Adjoint) -> Self::Output {
        *self - rhs
    }
}

impl Neg for Su3Adjoint {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Su3Adjoint::new(-self.data)
    }
}

impl Neg for &Su3Adjoint {
    type Output = Su3Adjoint;

    fn neg(self) -> Self::Output {
        Su3Adjoint::new(-self.data())
    }
}

/// Return the representation for the zero matrix.
impl Default for Su3Adjoint {
    /// Return the representation for the zero matrix.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// assert_eq!(
    ///     Su3Adjoint::default(),
    ///     Su3Adjoint::new_from_array([0_f64; 8])
    /// );
    /// ```
    fn default() -> Self {
        Su3Adjoint::new(Vector8::from_element(0_f64))
    }
}

impl std::fmt::Display for Su3Adjoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_matrix())
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
    fn index(&self, pos: usize) -> &Self::Output {
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
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
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
    pub const fn new(data: Vec<Matrix3<na::Complex<Real>>>) -> Self {
        Self { data }
    }

    /// Get the raw data.
    pub const fn data(&self) -> &Vec<Matrix3<na::Complex<Real>>> {
        &self.data
    }

    /// Get a mutable reference to the data
    pub fn data_mut(&mut self) -> &mut Vec<Matrix3<na::Complex<Real>>> {
        &mut self.data
    }

    /// Get the link_matrix as a Vec
    pub const fn as_vec(&self) -> &Vec<Matrix3<na::Complex<Real>>> {
        self.data()
    }

    /// Get the link_matrix as a Vec
    pub fn as_vec_mut(&mut self) -> &mut Vec<Matrix3<na::Complex<Real>>> {
        self.data_mut()
    }

    /// Get the link_matrix as a Vec
    pub fn as_slice(&self) -> &[Matrix3<na::Complex<Real>>] {
        self.data()
    }

    /// Get the link_matrix as a mut ref to a slice
    pub fn as_slice_mut(&mut self) -> &mut [Matrix3<na::Complex<Real>>] {
        &mut self.data
    }

    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data but slower than [`LinkMatrix::new_random_threaded`].
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclique};
    /// use rand::{rngs::StdRng, SeedableRng};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let lattice = LatticeCyclique::<4>::new(1_f64, 4)?;
    /// assert_eq!(
    ///     LinkMatrix::new_deterministe(&lattice, &mut rng_1),
    ///     LinkMatrix::new_deterministe(&lattice, &mut rng_2)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_deterministe<Rng: rand::Rng + ?Sized, const D: usize>(
        l: &LatticeCyclique<D>,
        rng: &mut Rng,
    ) -> Self {
        // l.get_links_space().map(|_| Su3Adjoint::random(rng, d).to_su3()).collect()
        // using a for loop imporves performance. ( probably because the vector is pre allocated).
        let mut data = Vec::with_capacity(l.get_number_of_canonical_links_space());
        for _ in l.get_links() {
            // the iterator *should* be in order
            let matrix = su3::get_random_su3(rng);
            data.push(matrix);
        }
        Self { data }
    }

    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reduced easily, In that case use [`LinkMatrix::new_random_threaded`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclique};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lattice = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrix::new_random_threaded(&lattice, 4)?;
    /// assert!(!links.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns [`ThreadError::ThreadNumberIncorect`] if `number_of_thread` is 0.
    pub fn new_random_threaded<const D: usize>(
        l: &LatticeCyclique<D>,
        number_of_thread: usize,
    ) -> Result<Self, ThreadError> {
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
        Ok(Self { data })
    }

    /// Create a cold configuration ( where the link matrices is set to the indentity).
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclique};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lattice = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrix::new_cold(&lattice);
    /// assert!(!links.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_cold<const D: usize>(l: &LatticeCyclique<D>) -> Self {
        Self {
            data: vec![CMatrix3::identity(); l.get_number_of_canonical_links_space()],
        }
    }

    /// get the link matrix associated to given link using the notation
    /// $`U_{-i}(x) = U^\dagger_{i}(x-i)`$
    pub fn get_matrix<const D: usize>(
        &self,
        link: &LatticeLink<D>,
        l: &LatticeCyclique<D>,
    ) -> Option<Matrix3<na::Complex<Real>>> {
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
    pub fn get_sij<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<Matrix3<na::Complex<Real>>> {
        let u_j = self.get_matrix(&LatticeLink::new(*point, *dir_j), lattice)?;
        let point_pj = lattice.add_point_direction(*point, dir_j);
        let u_i_p_j = self.get_matrix(&LatticeLink::new(point_pj, *dir_i), lattice)?;
        let point_pi = lattice.add_point_direction(*point, dir_i);
        let u_j_pi_d = self
            .get_matrix(&LatticeLink::new(point_pi, *dir_j), lattice)?
            .adjoint();
        Some(u_j * u_i_p_j * u_j_pi_d)
    }

    /// Get the plaquette $`P_{ij}(x) = U_i(x) S^\dagger_ij(x)`$.
    pub fn get_pij<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<Matrix3<na::Complex<Real>>> {
        let s_ij = self.get_sij(point, dir_i, dir_j, lattice)?;
        let u_i = self.get_matrix(&LatticeLink::new(*point, *dir_i), lattice)?;
        Some(u_i * s_ij.adjoint())
    }

    /// Take the average of the trace of all plaquettes
    #[allow(clippy::as_conversions)] // no try into for f64
    pub fn average_trace_plaquette<const D: usize>(
        &self,
        lattice: &LatticeCyclique<D>,
    ) -> Option<na::Complex<Real>> {
        if lattice.get_number_of_canonical_links_space() != self.len() {
            return None;
        }
        // the order does not matter as we sum
        let sum = lattice
            .get_points()
            .par_bridge()
            .map(|point| {
                Direction::positive_directions()
                    .iter()
                    .map(|dir_i| {
                        Direction::positive_directions()
                            .iter()
                            .filter(|dir_j| dir_i.index() < dir_j.index())
                            .map(|dir_j| {
                                self.get_pij(&point, dir_i, dir_j, lattice)
                                    .map(|el| el.trace())
                            })
                            .sum::<Option<na::Complex<Real>>>()
                    })
                    .sum::<Option<na::Complex<Real>>>()
            })
            .sum::<Option<na::Complex<Real>>>()?;
        let number_of_directions = (D * (D - 1)) / 2;
        let number_of_plaquette = (lattice.get_number_of_points() * number_of_directions) as f64;
        Some(sum / number_of_plaquette)
    }

    /// Get the clover, used for F_mu_nu tensor
    pub fn get_clover<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<CMatrix3> {
        Some(
            self.get_pij(point, dir_i, dir_j, lattice)?
                + self.get_pij(point, dir_j, &-dir_i, lattice)?
                + self.get_pij(point, &-dir_i, &-dir_j, lattice)?
                + self.get_pij(point, &-dir_j, dir_i, lattice)?,
        )
    }

    /// Get the `F^{ij}` tensor using the clover appropriation. The direction should be set to positive
    /// See arXive:1512.02374.
    // TODO negative directions
    pub fn get_f_mu_nu<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<CMatrix3> {
        let m = self.get_clover(point, dir_i, dir_j, lattice)?
            - self.get_clover(point, dir_j, dir_i, lattice)?;
        Some(m / Complex::from(8_f64 * lattice.size() * lattice.size()))
    }

    /// Get the chromomagentic field at a given point
    pub fn get_magnetic_field_vec<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<SVector<CMatrix3, D>> {
        let mut vec = SVector::<CMatrix3, D>::zeros();
        for dir in &Direction::<D>::positive_directions() {
            vec[dir.index()] = self.get_magnetic_field(point, dir, lattice)?;
        }
        Some(vec)
    }

    /// Get the chromomagentic field at a given point alongisde a given direction
    pub fn get_magnetic_field<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir: &Direction<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<CMatrix3> {
        let sum = Direction::<D>::positive_directions()
            .iter()
            .map(|dir_i| {
                Direction::<D>::positive_directions()
                    .iter()
                    .map(|dir_j| {
                        let f_mn = self.get_f_mu_nu(point, dir_i, dir_j, lattice)?;
                        let lc = Complex::from(
                            levi_civita(&[dir.index(), dir_i.index(), dir_j.index()]).to_f64(),
                        );
                        Some(f_mn * lc)
                    })
                    .sum::<Option<CMatrix3>>()
            })
            .sum::<Option<CMatrix3>>()?;
        Some(sum / Complex::new(0_f64, 2_f64))
    }

    /// Get the chromomagentic field at a given point alongisde a given direction given by lattice link
    pub fn get_magnetic_field_link<const D: usize>(
        &self,
        link: &LatticeLink<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<Matrix3<na::Complex<Real>>> {
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
    ///
    /// You can look at the example of [`super::simulation::LatticeStateDefault::normalize_link_matrices`]
    pub fn normalize(&mut self) {
        self.data.par_iter_mut().for_each(|el| {
            su3::orthonormalize_matrix_mut(el);
        });
    }

    /// Iter on the data
    pub fn iter(&self) -> impl Iterator<Item = &CMatrix3> + ExactSizeIterator + FusedIterator {
        self.data.iter()
    }

    /// Iter mutably on the data
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut CMatrix3> + ExactSizeIterator + FusedIterator {
        self.data.iter_mut()
    }
}

impl AsRef<Vec<CMatrix3>> for LinkMatrix {
    fn as_ref(&self) -> &Vec<CMatrix3> {
        self.as_vec()
    }
}

impl AsMut<Vec<CMatrix3>> for LinkMatrix {
    fn as_mut(&mut self) -> &mut Vec<CMatrix3> {
        self.as_vec_mut()
    }
}

impl AsRef<[CMatrix3]> for LinkMatrix {
    fn as_ref(&self) -> &[CMatrix3] {
        self.as_slice()
    }
}

impl AsMut<[CMatrix3]> for LinkMatrix {
    fn as_mut(&mut self) -> &mut [CMatrix3] {
        self.as_slice_mut()
    }
}

impl<'a> IntoIterator for &'a LinkMatrix {
    type IntoIter = <&'a Vec<CMatrix3> as IntoIterator>::IntoIter;
    type Item = &'a CMatrix3;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut LinkMatrix {
    type IntoIter = <&'a mut Vec<CMatrix3> as IntoIterator>::IntoIter;
    type Item = &'a mut CMatrix3;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl Index<usize> for LinkMatrix {
    type Output = CMatrix3;

    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl IndexMut<usize> for LinkMatrix {
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl<A> FromIterator<A> for LinkMatrix
where
    Vec<CMatrix3>: FromIterator<A>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = A>,
    {
        Self::new(Vec::from_iter(iter))
    }
}

impl<A> FromParallelIterator<A> for LinkMatrix
where
    Vec<CMatrix3>: FromParallelIterator<A>,
    A: Send,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = A>,
    {
        Self::new(Vec::from_par_iter(par_iter))
    }
}

impl<T> ParallelExtend<T> for LinkMatrix
where
    Vec<CMatrix3>: ParallelExtend<T>,
    T: Send,
{
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = T>,
    {
        self.data.par_extend(par_iter);
    }
}

impl<A> Extend<A> for LinkMatrix
where
    Vec<CMatrix3>: Extend<A>,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = A>,
    {
        self.data.extend(iter);
    }
}

/// Represent an electric field.
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct EField<const D: usize> {
    data: Vec<SVector<Su3Adjoint, D>>, // use a Vec<[Su3Adjoint; D]> instead ?
}

impl<const D: usize> EField<D> {
    /// Create a new "Electrical" field.
    pub fn new(data: Vec<SVector<Su3Adjoint, D>>) -> Self {
        Self { data }
    }

    /// Get the raw data.
    pub const fn data(&self) -> &Vec<SVector<Su3Adjoint, D>> {
        &self.data
    }

    /// Get a mut ref to the data data.
    pub fn data_mut(&mut self) -> &mut Vec<SVector<Su3Adjoint, D>> {
        &mut self.data
    }

    /// Get the e_field as a Vec of Vector of Su3Adjoint
    pub const fn as_vec(&self) -> &Vec<SVector<Su3Adjoint, D>> {
        self.data()
    }

    /// Get the e_field as a slice of Vector of Su3Adjoint
    pub fn as_slice(&self) -> &[SVector<Su3Adjoint, D>] {
        &self.data
    }

    /// Get the e_field as mut ref to slice of Vector of Su3Adjoint
    pub fn as_slice_mut(&mut self) -> &mut [SVector<Su3Adjoint, D>] {
        &mut self.data
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
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// let lattice = LatticeCyclique::<4>::new(1_f64, 4).unwrap();
    /// assert_eq!(
    ///     EField::new_deterministe(&lattice, &mut rng_1, &distribution),
    ///     EField::new_deterministe(&lattice, &mut rng_2, &distribution)
    /// );
    /// ```
    pub fn new_deterministe<Rng: rand::Rng + ?Sized>(
        l: &LatticeCyclique<D>,
        rng: &mut Rng,
        d: &impl rand_distr::Distribution<Real>,
    ) -> Self {
        let mut data = Vec::with_capacity(l.get_number_of_points());
        for _ in l.get_points() {
            // iterator *should* be ordoned
            data.push(SVector::<Su3Adjoint, D>::from_fn(|_, _| {
                Su3Adjoint::random(rng, d)
            }));
        }
        Self { data }
    }

    /// Single thread generation by seeding a new rng number.
    /// To create a seedable and reproducible set use [`EField::new_deterministe`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::EField, lattice::LatticeCyclique};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// let lattice = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let e_field = EField::new_random(&lattice, &distribution);
    /// assert!(!e_field.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_random(l: &LatticeCyclique<D>, d: &impl rand_distr::Distribution<Real>) -> Self {
        let mut rng = rand::thread_rng();
        EField::new_deterministe(l, &mut rng, d)
    }

    /// Create a new cold configuration for the electriccal field, i.e. all E ar set to 0.
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::EField, lattice::LatticeCyclique};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lattice = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let e_field = EField::new_cold(&lattice);
    /// assert!(!e_field.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_cold(l: &LatticeCyclique<D>) -> Self {
        let p1 = Su3Adjoint::new_from_array([0_f64; 8]);
        Self {
            data: vec![SVector::<Su3Adjoint, D>::from_element(p1); l.get_number_of_points()],
        }
    }

    /// Get `E(point) = [E_x(point), E_y(point), E_z(point)]`.
    pub fn get_e_vec(
        &self,
        point: &LatticePoint<D>,
        l: &LatticeCyclique<D>,
    ) -> Option<&SVector<Su3Adjoint, D>> {
        self.data.get(point.to_index(l))
    }

    /// Get `E_{dir}(point)`. The sign of the direction does not change the output. i.e.
    /// `E_{-dir}(point) = E_{dir}(point)`.
    pub fn get_e_field(
        &self,
        point: &LatticePoint<D>,
        dir: &Direction<D>,
        l: &LatticeCyclique<D>,
    ) -> Option<&Su3Adjoint> {
        let value = self.get_e_vec(point, l);
        match value {
            Some(vec) => vec.get(dir.index()),
            None => None,
        }
    }

    /// Returns wether there is no data
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the Gauss parameter `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    #[inline]
    pub fn get_gauss(
        &self,
        link_matrix: &LinkMatrix,
        point: &LatticePoint<D>,
        lattice: &LatticeCyclique<D>,
    ) -> Option<CMatrix3> {
        if lattice.get_number_of_points() != self.len()
            || lattice.get_number_of_canonical_links_space() != link_matrix.len()
        {
            return None;
        }
        Direction::positive_directions()
            .iter()
            .map(|dir| {
                let e_i = self.get_e_field(point, dir, lattice)?;
                let u_mi = link_matrix.get_matrix(&LatticeLink::new(*point, -*dir), lattice)?;
                let p_mi = lattice.add_point_direction(*point, &-dir);
                let e_m_i = self.get_e_field(&p_mi, dir, lattice)?;
                Some(e_i.to_matrix() - u_mi * e_m_i.to_matrix() * u_mi.adjoint())
            })
            .sum::<Option<CMatrix3>>()
    }

    /// Get the deviation from the Gauss law
    #[inline]
    pub fn get_gauss_sum_div(
        &self,
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
    ) -> Option<Real> {
        if lattice.get_number_of_points() != self.len()
            || lattice.get_number_of_canonical_links_space() != link_matrix.len()
        {
            return None;
        }
        lattice
            .get_points()
            .par_bridge()
            .map(|point| {
                self.get_gauss(link_matrix, &point, lattice).map(|el| {
                    (su3::GENERATORS.iter().copied().sum::<CMatrix3>() * el)
                        .trace()
                        .abs()
                })
            })
            .sum::<Option<Real>>()
    }

    /// project to that the gauss law is approximatively respected ( up to `f64::EPSILON * 10` per point).
    ///
    /// It is mainly use internally but can be use to correct numerical drit in simulations.
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::error::ImplementationError;
    /// use lattice_qcd_rs::integrator::SymplecticEulerRayon;
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeState, LatticeStateDefault, LatticeStateWithEField,
    ///     LatticeStateWithEFieldSyncDefault, SimulationStateSynchrone,
    /// };
    /// use rand::SeedableRng;
    ///
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
    /// let distribution =
    ///     rand::distributions::Uniform::from(-std::f64::consts::PI..std::f64::consts::PI);
    /// let mut state = LatticeStateWithEFieldSyncDefault::new_random_e_state(
    ///     LatticeStateDefault::<3>::new_deterministe(1_f64, 6_f64, 4, &mut rng).unwrap(),
    ///     &mut rng,
    /// ); // <- here internally when choosing radomly the EField it is projected.
    ///
    /// let integrator = SymplecticEulerRayon::default();
    /// for _ in 0..2 {
    ///     for _ in 0..10 {
    ///         state = state.simulate_sync(&integrator, 0.0001_f64)?;
    ///     }
    ///     // we correct the numberical drift of the EField.
    ///     let new_e_field = state
    ///         .e_field()
    ///         .project_to_gauss(state.link_matrix(), state.lattice())
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    ///     state.set_e_field(new_e_field);
    /// }
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    #[allow(clippy::as_conversions)] // no try into for f64
    #[inline]
    pub fn project_to_gauss(
        &self,
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
    ) -> Option<Self> {
        // TODO improve
        const NUMBER_FOR_LOOP: usize = 4;

        if lattice.get_number_of_points() != self.len()
            || lattice.get_number_of_canonical_links_space() != link_matrix.len()
        {
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
    fn project_to_gauss_step(
        &self,
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
    ) -> Self {
        /// see <https://arxiv.org/pdf/1512.02374.pdf>
        // TODO verify
        const K: na::Complex<f64> = na::Complex::new(0.12_f64, 0_f64);
        let data = lattice
            .get_points()
            .collect::<Vec<LatticePoint<D>>>()
            .par_iter()
            .map(|point| {
                let e = self.get_e_vec(point, lattice).unwrap();
                SVector::<_, D>::from_fn(|index_dir, _| {
                    let dir = Direction::<D>::positive_directions()[index_dir];
                    let u = link_matrix
                        .get_matrix(&LatticeLink::new(*point, dir), lattice)
                        .unwrap();
                    let gauss = self.get_gauss(link_matrix, point, lattice).unwrap();
                    let gauss_p = self
                        .get_gauss(
                            link_matrix,
                            &lattice.add_point_direction(*point, &dir),
                            lattice,
                        )
                        .unwrap();
                    Su3Adjoint::new(Vector8::from_fn(|index, _| {
                        2_f64
                            * (su3::GENERATORS[index]
                                * ((u * gauss * u.adjoint() * gauss_p - gauss) * K
                                    + su3::GENERATORS[index]
                                        * na::Complex::from(e[dir.index()][index])))
                            .trace()
                            .real()
                    }))
                })
            })
            .collect();
        Self::new(data)
    }
}

impl<const D: usize> AsRef<Vec<SVector<Su3Adjoint, D>>> for EField<D> {
    fn as_ref(&self) -> &Vec<SVector<Su3Adjoint, D>> {
        self.as_vec()
    }
}

impl<const D: usize> AsMut<Vec<SVector<Su3Adjoint, D>>> for EField<D> {
    fn as_mut(&mut self) -> &mut Vec<SVector<Su3Adjoint, D>> {
        self.data_mut()
    }
}

impl<const D: usize> AsRef<[SVector<Su3Adjoint, D>]> for EField<D> {
    fn as_ref(&self) -> &[SVector<Su3Adjoint, D>] {
        self.as_slice()
    }
}

impl<const D: usize> AsMut<[SVector<Su3Adjoint, D>]> for EField<D> {
    fn as_mut(&mut self) -> &mut [SVector<Su3Adjoint, D>] {
        self.as_slice_mut()
    }
}

impl<'a, const D: usize> IntoIterator for &'a EField<D> {
    type IntoIter = <&'a Vec<SVector<Su3Adjoint, D>> as IntoIterator>::IntoIter;
    type Item = &'a SVector<Su3Adjoint, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a mut EField<D> {
    type IntoIter = <&'a mut Vec<SVector<Su3Adjoint, D>> as IntoIterator>::IntoIter;
    type Item = &'a mut SVector<Su3Adjoint, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<const D: usize> Index<usize> for EField<D> {
    type Output = SVector<Su3Adjoint, D>;

    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl<const D: usize> IndexMut<usize> for EField<D> {
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl<A, const D: usize> FromIterator<A> for EField<D>
where
    Vec<SVector<Su3Adjoint, D>>: FromIterator<A>,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = A>,
    {
        Self::new(Vec::from_iter(iter))
    }
}

impl<A, const D: usize> FromParallelIterator<A> for EField<D>
where
    Vec<SVector<Su3Adjoint, D>>: FromParallelIterator<A>,
    A: Send,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = A>,
    {
        Self::new(Vec::from_par_iter(par_iter))
    }
}

impl<T, const D: usize> ParallelExtend<T> for EField<D>
where
    Vec<SVector<Su3Adjoint, D>>: ParallelExtend<T>,
    T: Send,
{
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = T>,
    {
        self.data.par_extend(par_iter);
    }
}

impl<A, const D: usize> Extend<A> for EField<D>
where
    Vec<SVector<Su3Adjoint, D>>: Extend<A>,
{
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = A>,
    {
        self.data.extend(iter);
    }
}

#[cfg(test)]
mod test {
    use approx::*;
    use rand::SeedableRng;

    use super::super::{lattice::*, Complex};
    use super::*;

    const EPSILON: f64 = 0.000_000_001_f64;
    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    fn test_get_e_field_pos_neg() {
        use super::super::lattice;

        let l = LatticeCyclique::new(1_f64, 4).unwrap();
        let e = EField::new(vec![SVector::<_, 4>::from([
            Su3Adjoint::from([1_f64; 8]),
            Su3Adjoint::from([2_f64; 8]),
            Su3Adjoint::from([3_f64; 8]),
            Su3Adjoint::from([2_f64; 8]),
        ])]);
        assert_eq!(
            e.get_e_field(
                &LatticePoint::new([0, 0, 0, 0].into()),
                &lattice::DirectionEnum::XPos.into(),
                &l
            ),
            e.get_e_field(
                &LatticePoint::new([0, 0, 0, 0].into()),
                &lattice::DirectionEnum::XNeg.into(),
                &l
            )
        );
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::op_ref)]
    fn test_su3_adj() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let d = rand::distributions::Uniform::from(-1_f64..1_f64);
        for _ in 0_u32..100_u32 {
            let v = Su3Adjoint::random(&mut rng, &d);
            let m = v.to_matrix();
            assert_abs_diff_eq!(
                v.trace_squared(),
                (m * m).trace().modulus(),
                epsilon = EPSILON
            );
            assert_eq_complex!(
                v.d(),
                nalgebra::Complex::new(0_f64, 1_f64) * m.determinant(),
                EPSILON
            );
            assert_eq_complex!(v.t(), -(m * m).trace() / Complex::from(2_f64), EPSILON);

            // ----
            let adj_1 = Su3Adjoint::default();
            let adj_2 = Su3Adjoint::new_from_array([1_f64; 8]);
            assert_eq!(adj_2, adj_2 + adj_1);
            assert_eq!(adj_2, &adj_2 + &adj_1);
            assert_eq!(adj_2, &adj_2 - &adj_1);
            assert_eq!(adj_1, &adj_2 - &adj_2);
            assert_eq!(adj_1, &adj_2 - adj_2);
            assert_eq!(adj_1, adj_2 - &adj_2);
            assert_eq!(adj_1, -&adj_1);
            let adj_3 = Su3Adjoint::new_from_array([2_f64; 8]);
            assert_eq!(adj_3, &adj_2 + &adj_2);
            assert_eq!(adj_3, &adj_2 * &2_f64);
            assert_eq!(adj_3, &2_f64 * &adj_2);
            assert_eq!(adj_3, 2_f64 * adj_2);
            assert_eq!(adj_3, &2_f64 * adj_2);
            assert_eq!(adj_3, 2_f64 * &adj_2);
            assert_eq!(adj_2, &adj_3 / &2_f64);
            assert_eq!(adj_2, &adj_3 / 2_f64);
            let mut adj_5 = Su3Adjoint::new_from_array([2_f64; 8]);
            adj_5 /= &2_f64;
            assert_eq!(adj_2, adj_5);
            let adj_4 = Su3Adjoint::new_from_array([-1_f64; 8]);
            assert_eq!(adj_2, -adj_4);
        }

        use crate::su3::su3_exp_r;
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let d = rand::distributions::Uniform::from(-1_f64..1_f64);
        for _ in 0_u32..10_u32 {
            let v = Su3Adjoint::random(&mut rng, &d);
            assert_eq!(su3_exp_r(v), v.exp());
        }
    }

    #[test]
    fn link_matrix() {
        let lattice = LatticeCyclique::<3>::new(1_f64, 4).unwrap();
        match LinkMatrix::new_random_threaded(&lattice, 0) {
            Err(ThreadError::ThreadNumberIncorect) => {}
            _ => panic!("unexpected ouptut"),
        }
        let link_s = LinkMatrix::new_random_threaded(&lattice, 2);
        assert!(link_s.is_ok());
        let mut link = link_s.unwrap();
        assert!(!link.is_empty());
        let l2 = LinkMatrix::new(vec![]);
        assert!(l2.is_empty());

        let _: &[_] = link.as_ref();
        let _: &Vec<_> = link.as_ref();
        let _: &mut [_] = link.as_mut();
        let _: &mut Vec<_> = link.as_mut();
        let _ = link.iter();
        let _ = link.iter_mut();
        let _ = (&link).into_iter();
        let _ = (&mut link).into_iter();
    }

    #[test]
    fn e_field() {
        let lattice = LatticeCyclique::<3>::new(1_f64, 4).unwrap();
        let e_field_s = LinkMatrix::new_random_threaded(&lattice, 2);
        assert!(e_field_s.is_ok());
        let mut e_field = e_field_s.unwrap();

        let _: &[_] = e_field.as_ref();
        let _: &Vec<_> = e_field.as_ref();
        let _: &mut [_] = e_field.as_mut();
        let _: &mut Vec<_> = e_field.as_mut();
        let _ = e_field.iter();
        let _ = e_field.iter_mut();
        let _ = (&e_field).into_iter();
        let _ = (&mut e_field).into_iter();
    }

    #[test]
    fn mangetic_field() {
        let lattice = LatticeCyclique::<3>::new(1_f64, 4).unwrap();
        let mut link_matrix = LinkMatrix::new_cold(&lattice);
        let point = LatticePoint::from([0, 0, 0]);
        let dir_x = Direction::new(0, true).unwrap();
        let dir_y = Direction::new(1, true).unwrap();
        let dir_z = Direction::new(2, true).unwrap();
        let clover = link_matrix
            .get_clover(&point, &dir_x, &dir_y, &lattice)
            .unwrap();
        assert_eq_matrix!(CMatrix3::identity() * Complex::from(4_f64), clover, EPSILON);
        let f = link_matrix
            .get_f_mu_nu(&point, &dir_x, &dir_y, &lattice)
            .unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), f, EPSILON);
        let b = link_matrix
            .get_magnetic_field(&point, &dir_x, &lattice)
            .unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        let b_vec = link_matrix
            .get_magnetic_field_vec(&point, &lattice)
            .unwrap();
        for i in &b_vec {
            assert_eq_matrix!(CMatrix3::zeros(), i, EPSILON);
        }
        // ---
        link_matrix[0] = CMatrix3::identity() * Complex::new(0_f64, 1_f64);
        let clover = link_matrix
            .get_clover(&point, &dir_x, &dir_y, &lattice)
            .unwrap();
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(2_f64, 0_f64),
            clover,
            EPSILON
        );
        let clover = link_matrix
            .get_clover(&point, &dir_y, &dir_x, &lattice)
            .unwrap();
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(2_f64, 0_f64),
            clover,
            EPSILON
        );
        let f = link_matrix
            .get_f_mu_nu(&point, &dir_x, &dir_y, &lattice)
            .unwrap();
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(0_f64, 0_f64),
            f,
            EPSILON
        );
        let b = link_matrix
            .get_magnetic_field(&point, &dir_x, &lattice)
            .unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        let b_vec = link_matrix
            .get_magnetic_field_vec(&point, &lattice)
            .unwrap();
        for i in &b_vec {
            assert_eq_matrix!(CMatrix3::zeros(), i, EPSILON);
        }
        assert_eq_matrix!(
            link_matrix
                .get_magnetic_field_link(&LatticeLink::new(point, dir_x), &lattice)
                .unwrap(),
            b,
            EPSILON
        );
        //--
        let mut link_matrix = LinkMatrix::new_cold(&lattice);
        let link = LatticeLinkCanonical::new([1, 0, 0].into(), dir_y).unwrap();
        link_matrix[link.to_index(&lattice)] = CMatrix3::identity() * Complex::new(0_f64, 1_f64);
        let clover = link_matrix
            .get_clover(&point, &dir_x, &dir_y, &lattice)
            .unwrap();
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(3_f64, 1_f64),
            clover,
            EPSILON
        );
        let clover = link_matrix
            .get_clover(&point, &dir_y, &dir_x, &lattice)
            .unwrap();
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(3_f64, -1_f64),
            clover,
            EPSILON
        );
        let f = link_matrix
            .get_f_mu_nu(&point, &dir_x, &dir_y, &lattice)
            .unwrap();
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(0_f64, 0.25_f64),
            f,
            EPSILON
        );
        let b = link_matrix
            .get_magnetic_field(&point, &dir_x, &lattice)
            .unwrap();
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        assert_eq_matrix!(
            link_matrix
                .get_magnetic_field_link(&LatticeLink::new(point, dir_x), &lattice)
                .unwrap(),
            b,
            EPSILON
        );
        let b = link_matrix
            .get_magnetic_field(&point, &dir_z, &lattice)
            .unwrap();
        assert_eq_matrix!(
            link_matrix
                .get_magnetic_field_link(&LatticeLink::new(point, dir_z), &lattice)
                .unwrap(),
            b,
            EPSILON
        );
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(0.25_f64, 0_f64),
            b,
            EPSILON
        );
        let b_2 = link_matrix
            .get_magnetic_field(&[4, 0, 0].into(), &dir_z, &lattice)
            .unwrap();
        assert_eq_matrix!(b, b_2, EPSILON);
        let b_vec = link_matrix
            .get_magnetic_field_vec(&point, &lattice)
            .unwrap();
        for (index, m) in b_vec.iter().enumerate() {
            if index == 2 {
                assert_eq_matrix!(m, b, EPSILON);
            }
            else {
                assert_eq_matrix!(CMatrix3::zeros(), m, EPSILON);
            }
        }
    }
}
