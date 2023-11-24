//! Represent the fields on the lattice.

use std::fmt::{self, Display};
use std::iter::{FromIterator, FusedIterator};
use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    vec::Vec,
};

use nalgebra::{ComplexField, Matrix3, SVector};
use rand_distr::Distribution;
use rayon::iter::FromParallelIterator;
use rayon::prelude::*;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    lattice::{Direction, LatticeCyclic, LatticeElementToIndex, LatticeLink, LatticePoint},
    su3,
    su3::GENERATORS,
    thread::{run_pool_parallel_vec_with_initializations_mutable, ThreadError},
    utils::levi_civita,
    CMatrix3, Complex, Real, Vector8, I,
};

/// Adjoint representation of SU(3), it is su(3) (i.e. the lie algebra).
/// See [`su3::GENERATORS`] to view the order of generators.
/// Note that the generators are normalize such that `Tr[T^a T^b] = \delta^{ab} / 2`
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Su3Adjoint {
    /// the underling representation
    data: Vector8<Real>,
}

#[allow(clippy::len_without_is_empty)]
impl Su3Adjoint {
    /// create a new [`Su3Adjoint`] representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// use lattice_qcd_rs::field::Su3Adjoint;
    /// use nalgebra::SVector;
    ///
    /// let su3 = Su3Adjoint::new(SVector::<f64, 8>::from_element(1_f64));
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(data: Vector8<Real>) -> Self {
        Self { data }
    }

    /// create a new [`Su3Adjoint`] representation where `M = M^a T^a`, where `T` are generators given in [`su3::GENERATORS`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// ```
    #[inline]
    #[must_use]
    pub fn new_from_array(data: [Real; 8]) -> Self {
        Self::new(Vector8::from(data))
    }

    /// get the data inside the [`Su3Adjoint`].
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &Vector8<Real> {
        &self.data
    }

    /// Get the su3 adjoint as a [`Vector8`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// #
    /// # let adj = Su3Adjoint::default();
    /// let max = adj.as_vector().max();
    /// let norm = adj.as_ref().norm();
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_vector(&self) -> &Vector8<Real> {
        self.data()
    }

    /// Get the su3 adjoint as mut ref to a [`Vector8`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::Su3Adjoint, Vector8};
    /// #
    /// let mut adj = Su3Adjoint::default(); // filled with 0.
    /// adj.as_vector_mut().apply(|el| *el += 1_f64);
    /// assert_eq!(adj.as_vector(), &Vector8::from_element(1_f64));
    ///
    /// adj.as_mut().set_magnitude(1_f64);
    ///
    /// let mut v = Vector8::from_element(1_f64);
    /// v.set_magnitude(1_f64);
    ///
    /// assert_eq!(adj.as_vector(), &v);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_vector_mut(&mut self) -> &mut Vector8<Real> {
        self.data_mut()
    }

    /// Returns the su(3) (Lie algebra) matrix.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::Su3Adjoint};
    /// let su3 = Su3Adjoint::new_from_array([1_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64]);
    /// assert_eq!(su3.to_matrix(), *lattice_qcd_rs::su3::GENERATORS[0]);
    /// ```
    // TODO self non consumed ?? passé en &self ? TODO bench
    #[inline]
    #[must_use]
    pub fn to_matrix(self) -> Matrix3<nalgebra::Complex<Real>> {
        self.data
            .into_iter()
            .enumerate()
            .map(|(pos, el)| *GENERATORS[pos] * nalgebra::Complex::<Real>::from(el))
            .sum()
    }

    /// Return the SU(3) matrix associated with this generator.
    /// Note that the function consume self.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::Su3Adjoint};
    /// let su3 = Su3Adjoint::new_from_array([1_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64, 0_f64]);
    /// assert_eq!(su3.to_su3().determinant(), nalgebra::Complex::from(1_f64));
    /// ```
    #[inline]
    #[must_use]
    pub fn to_su3(self) -> Matrix3<nalgebra::Complex<Real>> {
        // NOTE: should it consume ? the user can manually clone and there is use because
        // where the value is not necessary anymore.
        su3::su3_exp_i(self)
    }

    /// Return exp( T^a v^a) where v is self.
    /// Note that the function consume self.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Matrix3<nalgebra::Complex<Real>> {
        su3::su3_exp_r(self)
    }

    /// Create a new random SU3 adjoint.
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::field::Su3Adjoint;
    ///
    /// let mut rng = rand::thread_rng();
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// let su3 = Su3Adjoint::random(&mut rng, &distribution);
    /// ```
    #[inline]
    #[must_use]
    pub fn random<Rng, D>(rng: &mut Rng, d: &D) -> Self
    where
        Rng: rand::Rng + ?Sized,
        D: Distribution<Real> + ?Sized,
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
    #[must_use]
    pub fn trace_squared(&self) -> Real {
        // TODO investigate.
        self.data.iter().map(|el| el * el).sum::<Real>() / 2_f64
    }

    /// Return the t coeff `t = - 1/2 * Tr(X^2)`.
    /// If you are looking for the trace square use [`Self::trace_squared`] instead.
    ///
    /// It is used for [`su3::su3_exp_i`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::from([1_f64; 8]);
    /// let m = su3.to_matrix();
    /// assert_eq!(
    ///     nalgebra::Complex::new(su3.t(), 0_f64),
    ///     -nalgebra::Complex::from(0.5_f64) * (m * m).trace()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn t(&self) -> Real {
        -0.5_f64 * self.trace_squared()
    }

    /// Return the t coeff `d = i * det(X)`.
    /// Used for [`su3::su3_exp_i`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::from([1_f64; 8]);
    /// let m = su3.to_matrix();
    /// assert_eq!(
    ///     su3.d(),
    ///     nalgebra::Complex::new(0_f64, 1_f64) * m.determinant()
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn d(&self) -> nalgebra::Complex<Real> {
        self.to_matrix().determinant() * I
    }

    /// Return the number of data. This number is 8
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let su3 = Su3Adjoint::new(nalgebra::SVector::<f64, 8>::zeros());
    /// assert_eq!(su3.len(), 8);
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
        //8
    }

    /// Return the number of data. This number is 8
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::new(nalgebra::SVector::<f64, 8>::zeros());
    /// assert_eq!(Su3Adjoint::len_const(), su3.len());
    /// ```
    #[inline]
    #[must_use]
    pub const fn len_const() -> usize {
        8
    }

    /// Get an iterator over the elements.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let adj = Su3Adjoint::default();
    /// let sum_abs = adj.iter().map(|el| el.abs()).sum::<f64>();
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Real> + ExactSizeIterator + FusedIterator {
        self.data.iter()
    }

    /// Get an iterator over the mutable ref of elements.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// # let mut adj = Su3Adjoint::default();
    /// adj.iter_mut().for_each(|el| *el = *el / 2_f64);
    /// ```
    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Real> + ExactSizeIterator + FusedIterator {
        self.data.iter_mut()
    }

    /// Get a mutable reference over the data.
    #[inline]
    #[must_use]
    pub fn data_mut(&mut self) -> &mut Vector8<Real> {
        &mut self.data
    }
}

impl AsRef<Vector8<f64>> for Su3Adjoint {
    #[inline]
    fn as_ref(&self) -> &Vector8<f64> {
        self.as_vector()
    }
}

impl AsMut<Vector8<f64>> for Su3Adjoint {
    #[inline]
    fn as_mut(&mut self) -> &mut Vector8<f64> {
        self.as_vector_mut()
    }
}

impl<'a> IntoIterator for &'a Su3Adjoint {
    type IntoIter = <&'a Vector8<Real> as IntoIterator>::IntoIter;
    type Item = &'a Real;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Su3Adjoint {
    type IntoIter = <&'a mut Vector8<Real> as IntoIterator>::IntoIter;
    type Item = &'a mut Real;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl AddAssign for Su3Adjoint {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.data += other.data();
    }
}

impl Add<Self> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl Add<&Self> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl Add<Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn add(self, rhs: Su3Adjoint) -> Self::Output {
        rhs + self
    }
}

impl Add<&Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn add(self, rhs: &Su3Adjoint) -> Self::Output {
        self + *rhs
    }
}

impl MulAssign<f64> for Su3Adjoint {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.data *= rhs;
    }
}

impl Mul<Real> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: Real) -> Self::Output {
        self *= rhs;
        self
    }
}

impl Mul<&Real> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Real) -> Self::Output {
        self * (*rhs)
    }
}

impl Mul<Real> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn mul(self, rhs: Real) -> Self::Output {
        *self * rhs
    }
}

impl Mul<&Real> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn mul(self, rhs: &Real) -> Self::Output {
        *self * rhs
    }
}

impl Mul<Su3Adjoint> for Real {
    type Output = Su3Adjoint;

    #[inline]
    fn mul(self, rhs: Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Mul<&Su3Adjoint> for Real {
    type Output = Su3Adjoint;

    #[inline]
    fn mul(self, rhs: &Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Mul<Su3Adjoint> for &Real {
    type Output = Su3Adjoint;

    #[inline]
    fn mul(self, rhs: Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl Mul<&Su3Adjoint> for &Real {
    type Output = Su3Adjoint;

    #[inline]
    fn mul(self, rhs: &Su3Adjoint) -> Self::Output {
        rhs * self
    }
}

impl DivAssign<f64> for Su3Adjoint {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.data /= rhs;
    }
}

impl DivAssign<&f64> for Su3Adjoint {
    #[inline]
    fn div_assign(&mut self, rhs: &f64) {
        self.data /= *rhs;
    }
}

impl Div<Real> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: Real) -> Self::Output {
        self /= rhs;
        self
    }
}

impl Div<&Real> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Real) -> Self::Output {
        self / (*rhs)
    }
}

impl Div<Real> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn div(self, rhs: Real) -> Self::Output {
        *self / rhs
    }
}

impl Div<&Real> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn div(self, rhs: &Real) -> Self::Output {
        *self / rhs
    }
}

impl SubAssign for Su3Adjoint {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.data -= other.data();
    }
}

impl Sub<Self> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl Sub<&Self> for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl Sub<Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn sub(self, rhs: Su3Adjoint) -> Self::Output {
        rhs - self
    }
}

impl Sub<&Su3Adjoint> for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn sub(self, rhs: &Su3Adjoint) -> Self::Output {
        *self - rhs
    }
}

impl Neg for Su3Adjoint {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.data)
    }
}

impl Neg for &Su3Adjoint {
    type Output = Su3Adjoint;

    #[inline]
    fn neg(self) -> Self::Output {
        Su3Adjoint::new(-self.data())
    }
}

/// Return the representation for the zero matrix.
impl Default for Su3Adjoint {
    /// Return the representation for the zero matrix.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// assert_eq!(
    ///     Su3Adjoint::default(),
    ///     Su3Adjoint::new_from_array([0_f64; 8])
    /// );
    /// ```
    #[inline]
    fn default() -> Self {
        Self::new(Vector8::from_element(0_f64))
    }
}

impl Display for Su3Adjoint {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_matrix())
    }
}

impl Index<usize> for Su3Adjoint {
    type Output = Real;

    /// Get the element at position `pos`.
    ///
    /// # Panic
    /// Panics if the position is out of bound (greater or equal to 8).
    /// ```should_panic
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// let _ = su3[8];
    /// ```
    #[inline]
    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl IndexMut<usize> for Su3Adjoint {
    /// Get the element at position `pos`.
    ///
    /// # Panic
    /// Panics if the position is out of bound (greater or equal to 8).
    /// ```should_panic
    /// # use lattice_qcd_rs::field::Su3Adjoint;
    /// let mut su3 = Su3Adjoint::new_from_array([0_f64; 8]);
    /// su3[8] += 1_f64;
    /// ```
    #[inline]
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl From<Vector8<Real>> for Su3Adjoint {
    #[inline]
    fn from(v: Vector8<Real>) -> Self {
        Self::new(v)
    }
}

impl From<Su3Adjoint> for Vector8<Real> {
    #[inline]
    fn from(v: Su3Adjoint) -> Self {
        v.data
    }
}

impl From<&Su3Adjoint> for Vector8<Real> {
    #[inline]
    fn from(v: &Su3Adjoint) -> Self {
        v.data
    }
}

impl From<[Real; 8]> for Su3Adjoint {
    #[inline]
    fn from(v: [Real; 8]) -> Self {
        Self::new_from_array(v)
    }
}

/// Represents the link matrices
// TODO more doc
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LinkMatrix {
    data: Vec<Matrix3<nalgebra::Complex<Real>>>,
}

impl LinkMatrix {
    /// Create a new link matrix field from preexisting data.
    #[inline]
    #[must_use]
    pub const fn new(data: Vec<Matrix3<nalgebra::Complex<Real>>>) -> Self {
        Self { data }
    }

    /// Get the raw data.
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &Vec<Matrix3<nalgebra::Complex<Real>>> {
        &self.data
    }

    /// Get a mutable reference to the data
    #[inline]
    #[must_use]
    pub fn data_mut(&mut self) -> &mut Vec<Matrix3<nalgebra::Complex<Real>>> {
        &mut self.data
    }

    /// Get the `link_matrix` as a [`Vec`].
    #[inline]
    #[must_use]
    pub const fn as_vec(&self) -> &Vec<Matrix3<nalgebra::Complex<Real>>> {
        self.data()
    }

    /// Get the `link_matrix` as a mutable [`Vec`].
    #[inline]
    #[must_use]
    pub fn as_vec_mut(&mut self) -> &mut Vec<Matrix3<nalgebra::Complex<Real>>> {
        self.data_mut()
    }

    /// Get the `link_matrix` as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Matrix3<nalgebra::Complex<Real>>] {
        self.data()
    }

    /// Get the `link_matrix` as a mut ref to a slice
    #[inline]
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [Matrix3<nalgebra::Complex<Real>>] {
        &mut self.data
    }

    /// Single threaded generation with a given random number generator.
    /// useful to produce a deterministic set of data but slower than
    /// [`LinkMatrix::new_random_threaded`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclic};
    /// use rand::{rngs::StdRng, SeedableRng};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// assert_eq!(
    ///     LinkMatrix::new_determinist(&lattice, &mut rng_1),
    ///     LinkMatrix::new_determinist(&lattice, &mut rng_2)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn new_determinist<Rng: rand::Rng + ?Sized, const D: usize>(
        l: &LatticeCyclic<D>,
        rng: &mut Rng,
    ) -> Self {
        // l.get_links_space().map(|_| Su3Adjoint::random(rng, d).to_su3()).collect()
        // using a for loop improves performance. ( probably because the vector is pre allocated).
        let mut data = Vec::with_capacity(l.number_of_canonical_links_space());
        for _ in l.get_links() {
            // the iterator *should* be in order
            let matrix = su3::random_su3(rng);
            data.push(matrix);
        }
        Self { data }
    }

    /// Multi threaded generation of random data. Due to the non deterministic way threads
    /// operate a set cannot be reduced easily. If you want deterministic
    /// generation you can use [`LinkMatrix::new_random_threaded`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclic};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrix::new_random_threaded(&lattice, 4)?;
    /// assert!(!links.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// Returns [`ThreadError::ThreadNumberIncorrect`] if `number_of_thread` is 0.
    #[inline]
    pub fn new_random_threaded<const D: usize>(
        l: &LatticeCyclic<D>,
        number_of_thread: usize,
    ) -> Result<Self, ThreadError> {
        if number_of_thread == 0 {
            return Err(ThreadError::ThreadNumberIncorrect);
        } else if number_of_thread == 1 {
            let mut rng = rand::thread_rng();
            return Ok(Self::new_determinist(l, &mut rng));
        }
        let data = run_pool_parallel_vec_with_initializations_mutable(
            l.get_links(),
            &(),
            &|rng, _, ()| su3::random_su3(rng),
            rand::thread_rng,
            number_of_thread,
            l.number_of_canonical_links_space(),
            l,
            &CMatrix3::zeros(),
        )?;
        Ok(Self { data })
    }

    /// Create a cold configuration ( where the link matrices is set to the identity).
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::LinkMatrix, lattice::LatticeCyclic};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrix::new_cold(&lattice);
    /// assert!(!links.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn new_cold<const D: usize>(l: &LatticeCyclic<D>) -> Self {
        Self {
            data: vec![CMatrix3::identity(); l.number_of_canonical_links_space()],
        }
    }

    /// get the link matrix associated to given link using the notation
    /// $`U_{-i}(x) = U^\dagger_{i}(x-i)`$
    #[inline]
    #[must_use]
    pub fn matrix<const D: usize>(
        &self,
        link: &LatticeLink<D>,
        l: &LatticeCyclic<D>,
    ) -> Option<Matrix3<nalgebra::Complex<Real>>> {
        let link_c = l.into_canonical(*link);
        let matrix = self.data.get(link_c.to_index(l))?;
        if link.is_dir_negative() {
            // that means the the link was in the negative direction
            Some(matrix.adjoint())
        } else {
            Some(*matrix)
        }
    }

    /// Get $`S_ij(x) = U_j(x) U_i(x+j) U^\dagger_j(x+i)`$.
    #[allow(clippy::similar_names)]
    #[inline]
    #[must_use]
    pub fn sij<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<Matrix3<nalgebra::Complex<Real>>> {
        let u_j = self.matrix(&LatticeLink::new(*point, *dir_j), lattice)?;
        let point_pj = lattice.add_point_direction(*point, dir_j);
        let u_i_p_j = self.matrix(&LatticeLink::new(point_pj, *dir_i), lattice)?;
        let point_pi = lattice.add_point_direction(*point, dir_i);
        let u_j_pi_d = self
            .matrix(&LatticeLink::new(point_pi, *dir_j), lattice)?
            .adjoint();
        Some(u_j * u_i_p_j * u_j_pi_d)
    }

    /// Get the plaquette $`P_{ij}(x) = U_i(x) S^\dagger_ij(x)`$.
    #[inline]
    #[must_use]
    pub fn pij<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<Matrix3<nalgebra::Complex<Real>>> {
        let s_ij = self.sij(point, dir_i, dir_j, lattice)?;
        let u_i = self.matrix(&LatticeLink::new(*point, *dir_i), lattice)?;
        Some(u_i * s_ij.adjoint())
    }

    /// Take the average of the trace of all plaquettes
    #[inline]
    #[must_use]
    pub fn average_trace_plaquette<const D: usize>(
        &self,
        lattice: &LatticeCyclic<D>,
    ) -> Option<nalgebra::Complex<Real>> {
        if lattice.number_of_canonical_links_space() != self.len() {
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
                                self.pij(&point, dir_i, dir_j, lattice).map(|el| el.trace())
                            })
                            .sum::<Option<nalgebra::Complex<Real>>>()
                    })
                    .sum::<Option<nalgebra::Complex<Real>>>()
            })
            .sum::<Option<nalgebra::Complex<Real>>>()?;
        let number_of_directions = (D * (D - 1)) / 2;

        #[allow(clippy::as_conversions)] // no try into for f64
        #[allow(clippy::cast_precision_loss)]
        let number_of_plaquette = (lattice.number_of_points() * number_of_directions) as f64;
        Some(sum / number_of_plaquette)
    }

    /// Get the clover, used for `F_mu_nu` tensor
    #[inline]
    #[must_use]
    pub fn clover<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3> {
        Some(
            self.pij(point, dir_i, dir_j, lattice)?
                + self.pij(point, dir_j, &-dir_i, lattice)?
                + self.pij(point, &-dir_i, &-dir_j, lattice)?
                + self.pij(point, &-dir_j, dir_i, lattice)?,
        )
    }

    /// Get the `F^{ij}` tensor using the clover appropriation. The direction should be set to positive.
    /// See <https://arxiv.org/abs/1512.02374>.
    // TODO negative directions
    #[inline]
    #[must_use]
    pub fn f_mu_nu<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir_i: &Direction<D>,
        dir_j: &Direction<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3> {
        let m = self.clover(point, dir_i, dir_j, lattice)?
            - self.clover(point, dir_j, dir_i, lattice)?;
        Some(m / Complex::from(8_f64 * lattice.size() * lattice.size()))
    }

    /// Get the chromomagnetic field at a given point.
    #[inline]
    #[must_use]
    pub fn magnetic_field_vec<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<SVector<CMatrix3, D>> {
        let mut vec = SVector::<CMatrix3, D>::zeros();
        for dir in &Direction::<D>::positive_directions() {
            vec[dir.index()] = self.magnetic_field(point, dir, lattice)?;
        }
        Some(vec)
    }

    /// Get the chromomagnetic field at a given point alongside a given direction.
    #[inline]
    #[must_use]
    pub fn magnetic_field<const D: usize>(
        &self,
        point: &LatticePoint<D>,
        dir: &Direction<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3> {
        let sum = Direction::<D>::positive_directions()
            .iter()
            .map(|dir_i| {
                Direction::<D>::positive_directions()
                    .iter()
                    .map(|dir_j| {
                        let f_mn = self.f_mu_nu(point, dir_i, dir_j, lattice)?;
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

    /// Get the chromomagnetic field at a given point alongside a given direction given by lattice link.
    #[inline]
    #[must_use]
    pub fn magnetic_field_link<const D: usize>(
        &self,
        link: &LatticeLink<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<Matrix3<nalgebra::Complex<Real>>> {
        self.magnetic_field(link.pos(), link.dir(), lattice)
    }

    /// Return the number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns wether the there is no data.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Correct the numerical drift, reprojecting all the matrices to SU(3).
    ///
    /// You can look at the example of [`super::simulation::LatticeStateDefault::normalize_link_matrices`]
    #[inline]
    pub fn normalize(&mut self) {
        self.data.par_iter_mut().for_each(|el| {
            su3::orthonormalize_matrix_mut(el);
        });
    }

    /// Iter on the data.
    #[inline]
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = &CMatrix3> + ExactSizeIterator + FusedIterator + DoubleEndedIterator
    {
        self.into_iter()
    }

    /// Iter mutably on the data.
    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut CMatrix3> + ExactSizeIterator + FusedIterator + DoubleEndedIterator
    {
        self.into_iter()
    }
}

impl AsRef<Vec<CMatrix3>> for LinkMatrix {
    #[inline]
    fn as_ref(&self) -> &Vec<CMatrix3> {
        self.as_vec()
    }
}

impl AsMut<Vec<CMatrix3>> for LinkMatrix {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<CMatrix3> {
        self.as_vec_mut()
    }
}

impl AsRef<[CMatrix3]> for LinkMatrix {
    #[inline]
    fn as_ref(&self) -> &[CMatrix3] {
        self.as_slice()
    }
}

impl AsMut<[CMatrix3]> for LinkMatrix {
    #[inline]
    fn as_mut(&mut self) -> &mut [CMatrix3] {
        self.as_slice_mut()
    }
}

impl<'a> IntoIterator for &'a LinkMatrix {
    type IntoIter = <&'a Vec<CMatrix3> as IntoIterator>::IntoIter;
    type Item = &'a CMatrix3;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut LinkMatrix {
    type IntoIter = <&'a mut Vec<CMatrix3> as IntoIterator>::IntoIter;
    type Item = &'a mut CMatrix3;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl Index<usize> for LinkMatrix {
    type Output = CMatrix3;

    #[inline]
    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl IndexMut<usize> for LinkMatrix {
    #[inline]
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl<A> FromIterator<A> for LinkMatrix
where
    Vec<CMatrix3>: FromIterator<A>,
{
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    #[must_use]
    pub fn new(data: Vec<SVector<Su3Adjoint, D>>) -> Self {
        Self { data }
    }

    /// Get the raw data.
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &Vec<SVector<Su3Adjoint, D>> {
        &self.data
    }

    /// Get a mut ref to the data data.
    #[inline]
    #[must_use]
    pub fn data_mut(&mut self) -> &mut Vec<SVector<Su3Adjoint, D>> {
        &mut self.data
    }

    /// Get the `e_field` as a Vec of Vector of [`Su3Adjoint`]
    #[inline]
    #[must_use]
    pub const fn as_vec(&self) -> &Vec<SVector<Su3Adjoint, D>> {
        self.data()
    }

    /// Get the `e_field` as a slice of Vector of [`Su3Adjoint`]
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[SVector<Su3Adjoint, D>] {
        &self.data
    }

    /// Get the `e_field` as mut ref to slice of Vector of [`Su3Adjoint`]
    #[inline]
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [SVector<Su3Adjoint, D>] {
        &mut self.data
    }

    /// Return the number of elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Single threaded generation with a given random number generator.
    /// useful to reproduce a set of data.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{field::EField, lattice::LatticeCyclic};
    /// # fn main () -> Result<(), Box<dyn std::error::Error>> {
    /// use rand::{rngs::StdRng, SeedableRng};
    ///
    /// let mut rng_1 = StdRng::seed_from_u64(0);
    /// let mut rng_2 = StdRng::seed_from_u64(0);
    /// // They have the same seed and should generate the same numbers
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// assert_eq!(
    ///     EField::new_determinist(&lattice, &mut rng_1, &distribution),
    ///     EField::new_determinist(&lattice, &mut rng_2, &distribution)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn new_determinist<Rng, Dist>(l: &LatticeCyclic<D>, rng: &mut Rng, d: &Dist) -> Self
    where
        Rng: rand::Rng + ?Sized,
        Dist: Distribution<Real> + ?Sized,
    {
        let mut data = Vec::with_capacity(l.number_of_points());
        for _ in l.get_points() {
            // iterator *should* be ordered
            data.push(SVector::<Su3Adjoint, D>::from_fn(|_, _| {
                Su3Adjoint::random(rng, d)
            }));
        }
        Self { data }
    }

    /// Single thread generation by seeding a new rng number.
    /// To create a seedable and reproducible set use [`EField::new_determinist`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::EField, lattice::LatticeCyclic};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
    /// let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
    /// let e_field = EField::new_random(&lattice, &distribution);
    /// assert!(!e_field.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn new_random<Dist>(l: &LatticeCyclic<D>, d: &Dist) -> Self
    where
        Dist: Distribution<Real> + ?Sized,
    {
        let mut rng = rand::thread_rng();
        Self::new_determinist(l, &mut rng, d)
    }

    /// Create a new cold configuration for the electrical field, i.e. all E ar set to 0.
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{field::EField, lattice::LatticeCyclic};
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
    /// let e_field = EField::new_cold(&lattice);
    /// assert!(!e_field.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn new_cold(l: &LatticeCyclic<D>) -> Self {
        let p1 = Su3Adjoint::new_from_array([0_f64; 8]);
        Self {
            data: vec![SVector::<Su3Adjoint, D>::from_element(p1); l.number_of_points()],
        }
    }

    /// Get `E(point) = [E_x(point), E_y(point), E_z(point)]`.
    #[inline]
    #[must_use]
    pub fn e_vec(
        &self,
        point: &LatticePoint<D>,
        l: &LatticeCyclic<D>,
    ) -> Option<&SVector<Su3Adjoint, D>> {
        self.data.get(point.to_index(l))
    }

    /// Get `E_{dir}(point)`. The sign of the direction does not change the output. i.e.
    /// `E_{-dir}(point) = E_{dir}(point)`.
    #[inline]
    #[must_use]
    pub fn e_field(
        &self,
        point: &LatticePoint<D>,
        dir: &Direction<D>,
        l: &LatticeCyclic<D>,
    ) -> Option<&Su3Adjoint> {
        let value = self.e_vec(point, l);
        value.and_then(|vec| vec.get(dir.index()))
    }

    /// Returns wether there is no data
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the Gauss parameter `G(x) = \sum_i E_i(x) - U_{-i}(x) E_i(x - i) U^\dagger_{-i}(x)`.
    #[inline]
    #[must_use]
    pub fn gauss(
        &self,
        link_matrix: &LinkMatrix,
        point: &LatticePoint<D>,
        lattice: &LatticeCyclic<D>,
    ) -> Option<CMatrix3> {
        if lattice.number_of_points() != self.len()
            || lattice.number_of_canonical_links_space() != link_matrix.len()
        {
            return None;
        }
        Direction::positive_directions()
            .iter()
            .map(|dir| {
                let e_i = self.e_field(point, dir, lattice)?;
                let u_mi = link_matrix.matrix(&LatticeLink::new(*point, -*dir), lattice)?;
                let p_mi = lattice.add_point_direction(*point, &-dir);
                let e_m_i = self.e_field(&p_mi, dir, lattice)?;
                Some(e_i.to_matrix() - u_mi * e_m_i.to_matrix() * u_mi.adjoint())
            })
            .sum::<Option<CMatrix3>>()
    }

    /// Get the deviation from the Gauss law
    #[inline]
    #[must_use]
    pub fn gauss_sum_div(
        &self,
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclic<D>,
    ) -> Option<Real> {
        if lattice.number_of_points() != self.len()
            || lattice.number_of_canonical_links_space() != link_matrix.len()
        {
            return None;
        }
        lattice
            .get_points()
            .par_bridge()
            .map(|point| {
                self.gauss(link_matrix, &point, lattice).map(|el| {
                    (su3::GENERATORS.iter().copied().sum::<CMatrix3>() * el)
                        .trace()
                        .abs()
                })
            })
            .sum::<Option<Real>>()
    }

    /// project to that the gauss law is approximately respected ( up to `f64::EPSILON * 10` per point).
    ///
    /// It is mainly use internally but can be use to correct numerical drift in simulations.
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::error::ImplementationError;
    /// use lattice_qcd_rs::integrator::SymplecticEulerRayon;
    /// use lattice_qcd_rs::simulation::{
    ///     LatticeState, LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField,
    ///     SimulationStateSynchronous,
    /// };
    /// use rand::SeedableRng;
    ///
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
    /// let distribution =
    ///     rand::distributions::Uniform::from(-std::f64::consts::PI..std::f64::consts::PI);
    /// let mut state = LatticeStateEFSyncDefault::new_random_e_state(
    ///     LatticeStateDefault::<3>::new_determinist(1_f64, 6_f64, 4, &mut rng)?,
    ///     &mut rng,
    /// ); // <- here internally when choosing randomly the EField it is projected.
    ///
    /// let integrator = SymplecticEulerRayon::default();
    /// for _ in 0..2 {
    ///     for _ in 0..10 {
    ///         state = state.simulate_sync(&integrator, 0.0001_f64)?;
    ///     }
    ///     // we correct the numerical drift of the EField.
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
    #[allow(clippy::cast_precision_loss)]
    #[inline]
    #[must_use]
    pub fn project_to_gauss(
        &self,
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclic<D>,
    ) -> Option<Self> {
        // TODO improve
        const NUMBER_FOR_LOOP: usize = 4;

        if lattice.number_of_points() != self.len()
            || lattice.number_of_canonical_links_space() != link_matrix.len()
        {
            return None;
        }
        let mut return_val = self.project_to_gauss_step(link_matrix, lattice);
        loop {
            let val_dif = return_val.gauss_sum_div(link_matrix, lattice)?;
            //println!("diff : {}", val_dif);
            if val_dif.is_nan() {
                return None;
            }
            if val_dif <= f64::EPSILON * (lattice.number_of_points() * 4 * 8 * 10) as f64 {
                break;
            }
            for _ in 0_usize..NUMBER_FOR_LOOP {
                return_val = return_val.project_to_gauss_step(link_matrix, lattice);
                //println!("{}", return_val[0][0][0]);
            }
        }
        Some(return_val)
    }

    /// Done one step to project to gauss law.
    ///
    /// # Panic
    /// panics if the link matrix and lattice is not of the correct size.
    #[inline] // REVIEW
    fn project_to_gauss_step(&self, link_matrix: &LinkMatrix, lattice: &LatticeCyclic<D>) -> Self {
        /// see <https://arxiv.org/pdf/1512.02374.pdf>
        // TODO verify
        const K: nalgebra::Complex<f64> = nalgebra::Complex::new(0.12_f64, 0_f64);
        let data = lattice
            .get_points()
            .par_iter()
            .map(|point| {
                let e = self.e_vec(&point, lattice).expect("e vec not founc");
                SVector::<_, D>::from_fn(|index_dir, _| {
                    let dir = Direction::<D>::positive_directions()[index_dir];
                    let u = link_matrix
                        .matrix(&LatticeLink::new(point, dir), lattice)
                        .expect("matrix not found");
                    let gauss = self
                        .gauss(link_matrix, &point, lattice)
                        .expect("gauss not found");
                    let gauss_p = self
                        .gauss(
                            link_matrix,
                            &lattice.add_point_direction(point, &dir),
                            lattice,
                        )
                        .expect("gauss not found");
                    Su3Adjoint::new(Vector8::from_fn(|index, _| {
                        2_f64
                            * (su3::GENERATORS[index]
                                * ((u * gauss * u.adjoint() * gauss_p - gauss) * K
                                    + su3::GENERATORS[index]
                                        * nalgebra::Complex::from(e[dir.index()][index])))
                            .trace()
                            .real()
                    }))
                })
            })
            .collect();
        Self::new(data)
    }

    // TODO test
    /// Gives an iterator over the [`SVector`] in the order they are stored in
    /// the underlying vector.
    #[inline]
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = &SVector<Su3Adjoint, D>>
           + ExactSizeIterator
           + FusedIterator
           + DoubleEndedIterator {
        self.into_iter()
    }

    /// Gives an iterator over a mutable reference of the [`SVector`] in the order
    /// they are stored in the underlying vector.
    #[inline]
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut SVector<Su3Adjoint, D>>
           + ExactSizeIterator
           + FusedIterator
           + DoubleEndedIterator {
        self.into_iter()
    }
}

impl<const D: usize> AsRef<Vec<SVector<Su3Adjoint, D>>> for EField<D> {
    #[inline]
    fn as_ref(&self) -> &Vec<SVector<Su3Adjoint, D>> {
        self.as_vec()
    }
}

impl<const D: usize> AsMut<Vec<SVector<Su3Adjoint, D>>> for EField<D> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec<SVector<Su3Adjoint, D>> {
        self.data_mut()
    }
}

impl<const D: usize> AsRef<[SVector<Su3Adjoint, D>]> for EField<D> {
    #[inline]
    fn as_ref(&self) -> &[SVector<Su3Adjoint, D>] {
        self.as_slice()
    }
}

impl<const D: usize> AsMut<[SVector<Su3Adjoint, D>]> for EField<D> {
    #[inline]
    fn as_mut(&mut self) -> &mut [SVector<Su3Adjoint, D>] {
        self.as_slice_mut()
    }
}

// TODO into iter and par iter

impl<'a, const D: usize> IntoIterator for &'a EField<D> {
    type IntoIter = <&'a Vec<SVector<Su3Adjoint, D>> as IntoIterator>::IntoIter;
    type Item = &'a SVector<Su3Adjoint, D>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a mut EField<D> {
    type IntoIter = <&'a mut Vec<SVector<Su3Adjoint, D>> as IntoIterator>::IntoIter;
    type Item = &'a mut SVector<Su3Adjoint, D>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<const D: usize> Index<usize> for EField<D> {
    type Output = SVector<Su3Adjoint, D>;

    #[inline]
    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl<const D: usize> IndexMut<usize> for EField<D> {
    #[inline]
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl<A, const D: usize> FromIterator<A> for EField<D>
where
    Vec<SVector<Su3Adjoint, D>>: FromIterator<A>,
{
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = A>,
    {
        self.data.extend(iter);
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use approx::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::Uniform;

    use super::super::{lattice::*, Complex};
    use super::*;
    use crate::error::{ImplementationError, LatticeInitializationError};
    use crate::su3::su3_exp_r;

    const EPSILON: f64 = 0.000_000_001_f64;
    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    fn test_get_e_field_pos_neg() -> Result<(), LatticeInitializationError> {
        use super::super::lattice;

        let l = LatticeCyclic::new(1_f64, 4)?;
        let e = EField::new(vec![SVector::<_, 4>::from([
            Su3Adjoint::from([1_f64; 8]),
            Su3Adjoint::from([2_f64; 8]),
            Su3Adjoint::from([3_f64; 8]),
            Su3Adjoint::from([2_f64; 8]),
        ])]);
        assert_eq!(
            e.e_field(
                &LatticePoint::new([0, 0, 0, 0].into()),
                &lattice::DirectionEnum::XPos.into(),
                &l
            ),
            e.e_field(
                &LatticePoint::new([0, 0, 0, 0].into()),
                &lattice::DirectionEnum::XNeg.into(),
                &l
            )
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::eq_op)]
    #[allow(clippy::op_ref)]
    fn test_su3_adj() {
        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        let d = Uniform::from(-1_f64..1_f64);
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

        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        let d = Uniform::from(-1_f64..1_f64);
        for _ in 0_u32..10_u32 {
            let v = Su3Adjoint::random(&mut rng, &d);
            assert_eq!(su3_exp_r(v), v.exp());
        }
    }

    // FIXME
    #[test]
    fn link_matrix() -> Result<(), Box<dyn Error>> {
        let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
        match LinkMatrix::new_random_threaded(&lattice, 0) {
            Err(ThreadError::ThreadNumberIncorrect) => {}
            _ => panic!("unexpected output"),
        }
        let link_s = LinkMatrix::new_random_threaded(&lattice, 2);
        assert!(link_s.is_ok());
        let mut link = link_s?;
        assert!(!link.is_empty());
        let l2 = LinkMatrix::new(vec![]);
        assert!(l2.is_empty());

        let _: &[_] = link.as_ref();
        let _: &Vec<_> = link.as_ref();
        let _: &mut [_] = link.as_mut();
        let _: &mut Vec<_> = link.as_mut();
        #[allow(clippy::let_underscore_must_use)]
        #[allow(clippy::let_underscore_untyped)]
        {
            let _ = link.iter();
            let _ = link.iter_mut();
            let _ = (&link).into_iter();
            let _ = (&mut link).into_iter();
        }
        Ok(())
    }

    // FIXME
    #[test]
    fn e_field() -> Result<(), Box<dyn Error>> {
        let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
        let e_field_s = LinkMatrix::new_random_threaded(&lattice, 2);
        assert!(e_field_s.is_ok());
        let mut e_field = e_field_s?;

        let _: &[_] = e_field.as_ref();
        let _: &Vec<_> = e_field.as_ref();
        let _: &mut [_] = e_field.as_mut();
        let _: &mut Vec<_> = e_field.as_mut();
        #[allow(clippy::let_underscore_must_use)]
        #[allow(clippy::let_underscore_untyped)]
        {
            let _ = e_field.iter();
            let _ = e_field.iter_mut();
            let _ = (&e_field).into_iter();
            let _ = (&mut e_field).into_iter();
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    #[test]
    fn magnetic_field() -> Result<(), Box<dyn Error>> {
        let lattice = LatticeCyclic::<3>::new(1_f64, 4)?;
        let mut link_matrix = LinkMatrix::new_cold(&lattice);
        let point = LatticePoint::from([0, 0, 0]);
        let dir_x = Direction::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        let dir_y = Direction::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        let dir_z = Direction::new(2, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        let clover = link_matrix
            .clover(&point, &dir_x, &dir_y, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(CMatrix3::identity() * Complex::from(4_f64), clover, EPSILON);
        let f = link_matrix
            .f_mu_nu(&point, &dir_x, &dir_y, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(CMatrix3::zeros(), f, EPSILON);
        let b = link_matrix
            .magnetic_field(&point, &dir_x, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        let b_vec = link_matrix
            .magnetic_field_vec(&point, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        for i in &b_vec {
            assert_eq_matrix!(CMatrix3::zeros(), i, EPSILON);
        }
        // ---
        link_matrix[0] = CMatrix3::identity() * Complex::new(0_f64, 1_f64);
        let clover = link_matrix
            .clover(&point, &dir_x, &dir_y, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(2_f64, 0_f64),
            clover,
            EPSILON
        );
        let clover = link_matrix
            .clover(&point, &dir_y, &dir_x, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(2_f64, 0_f64),
            clover,
            EPSILON
        );
        let f = link_matrix
            .f_mu_nu(&point, &dir_x, &dir_y, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(0_f64, 0_f64),
            f,
            EPSILON
        );
        let b = link_matrix
            .magnetic_field(&point, &dir_x, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        let b_vec = link_matrix
            .magnetic_field_vec(&point, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        for i in &b_vec {
            assert_eq_matrix!(CMatrix3::zeros(), i, EPSILON);
        }
        assert_eq_matrix!(
            link_matrix
                .magnetic_field_link(&LatticeLink::new(point, dir_x), &lattice)
                .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
            b,
            EPSILON
        );
        //--
        let mut link_matrix = LinkMatrix::new_cold(&lattice);
        let link = LatticeLinkCanonical::new([1, 0, 0].into(), dir_y)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        link_matrix[link.to_index(&lattice)] = CMatrix3::identity() * Complex::new(0_f64, 1_f64);
        let clover = link_matrix
            .clover(&point, &dir_x, &dir_y, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(3_f64, 1_f64),
            clover,
            EPSILON
        );
        let clover = link_matrix
            .clover(&point, &dir_y, &dir_x, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(3_f64, -1_f64),
            clover,
            EPSILON
        );
        let f = link_matrix
            .f_mu_nu(&point, &dir_x, &dir_y, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(0_f64, 0.25_f64),
            f,
            EPSILON
        );
        let b = link_matrix
            .magnetic_field(&point, &dir_x, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(CMatrix3::zeros(), b, EPSILON);
        assert_eq_matrix!(
            link_matrix
                .magnetic_field_link(&LatticeLink::new(point, dir_x), &lattice)
                .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
            b,
            EPSILON
        );
        let b = link_matrix
            .magnetic_field(&point, &dir_z, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(
            link_matrix
                .magnetic_field_link(&LatticeLink::new(point, dir_z), &lattice)
                .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
            b,
            EPSILON
        );
        assert_eq_matrix!(
            CMatrix3::identity() * Complex::new(0.25_f64, 0_f64),
            b,
            EPSILON
        );
        let b_2 = link_matrix
            .magnetic_field(&[4, 0, 0].into(), &dir_z, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(b, b_2, EPSILON);
        let b_vec = link_matrix
            .magnetic_field_vec(&point, &lattice)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        for (index, m) in b_vec.iter().enumerate() {
            if index == 2 {
                assert_eq_matrix!(m, b, EPSILON);
            } else {
                assert_eq_matrix!(CMatrix3::zeros(), m, EPSILON);
            }
        }

        Ok(())
    }

    #[test]
    fn test_len() {
        let su3 = Su3Adjoint::new(nalgebra::SVector::<f64, 8>::zeros());
        assert_eq!(Su3Adjoint::len_const(), su3.len());
    }
}
