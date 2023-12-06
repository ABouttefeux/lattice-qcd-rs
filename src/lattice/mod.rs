//! Defines lattices and lattice component.
//!
//! [`LatticeCyclic`] is the structure that encode the lattice information like
//! the lattice spacing, the number of point and the dimension.
//! It is used to do operation on [`LatticePoint`], [`LatticeLink`] and
//! [`LatticeLinkCanonical`].
//! Or to get an iterator over these elements.
//!
//! [`LatticePoint`], [`LatticeLink`] and [`LatticeLinkCanonical`] are elements on the lattice.
//! They encode where a field element is situated.

mod iterator;
mod lattice_cyclic;

use std::cmp::Ordering;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::{self, Display};
use std::iter::FusedIterator;
use std::ops::{Index, IndexMut, Neg};

use lattice_qcd_rs_procedural_macro::{implement_direction_from, implement_direction_list};
use nalgebra::{SVector, Vector4};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::Sealed;

// TODO remove IteratorElement from public interface ?
pub use self::iterator::{
    IteratorDirection, IteratorElement, IteratorLatticeLinkCanonical, IteratorLatticePoint,
    LatticeIterator, LatticeParIter, ParIterLatticeLinkCanonical, ParIterLatticePoint,
};
pub use self::lattice_cyclic::LatticeCyclic;
use super::Real;
use crate::private::Sealed;

/// Represents point on a (any) lattice.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Hash, Sealed)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticePoint<const D: usize> {
    data: nalgebra::SVector<usize, D>,
}

impl<const D: usize> LatticePoint<D> {
    /// Create a new lattice point.
    ///
    /// It can be outside a lattice.
    #[must_use]
    #[inline]
    pub const fn new(data: SVector<usize, D>) -> Self {
        Self { data }
    }

    /// Create a point at the origin
    #[must_use]
    #[inline]
    pub fn new_zero() -> Self {
        Self {
            data: SVector::zeros(),
        }
    }

    /// Create a point using the closure generate elements with the index as input.
    ///
    /// See [`nalgebra::base::Matrix::from_fn`].
    #[must_use]
    #[inline]
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> usize,
    {
        Self::new(SVector::from_fn(|index, _| f(index)))
    }

    /// Number of elements in [`LatticePoint`]. This is `D`.
    #[allow(clippy::unused_self)]
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        // this is in order to have a const function.
        // we could have called self.data.len()
        D
    }

    /// Return if [`LatticePoint`] contain no data. True when the dimension is 0, false otherwise.
    #[allow(clippy::unused_self)]
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        D == 0
    }

    /// Get an iterator on the data.
    #[inline]
    #[allow(clippy::implied_bounds_in_impls)] // no way to determine the Item of iterator otherwise
    pub fn iter(&self) -> impl Iterator<Item = &usize> + ExactSizeIterator + FusedIterator {
        self.data.iter()
    }

    /// Get an iterator on the data as mutable.
    #[inline]
    #[allow(clippy::implied_bounds_in_impls)] // no way to determine the Item of iterator otherwise
    pub fn iter_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut usize> + ExactSizeIterator + FusedIterator {
        self.data.iter_mut()
    }

    /// Get the point as as [`nalgebra::SVector<usize, D>`]
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// #
    /// # let point = LatticePoint::<4>::default();
    /// let max = point.as_svector().max();
    /// let min = point.as_ref().min();
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_svector(&self) -> &SVector<usize, D> {
        &self.data
    }

    /// Get the point as a mut ref to [`nalgebra::SVector<usize, D>`]
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// #
    /// # let mut point = LatticePoint::<4>::new_zero();
    /// point.as_svector_mut()[2] = 2;
    /// point.as_mut()[0] = 1;
    /// ```
    #[must_use]
    #[inline]
    pub fn as_svector_mut(&mut self) -> &mut SVector<usize, D> {
        &mut self.data
    }

    #[inline]
    #[must_use]
    fn index_to_point(lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
        (index < lattice.number_of_points()).then(|| {
            Self::new(SVector::from_fn(|i, _| {
                (index / lattice.dim().pow(i.try_into().expect("conversion error"))) % lattice.dim()
            }))
        })
    }
}

impl<const D: usize> Default for LatticePoint<D> {
    #[inline]
    fn default() -> Self {
        Self::new_zero()
    }
}

impl<const D: usize> Display for LatticePoint<D> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<'a, const D: usize> IntoIterator for &'a LatticePoint<D> {
    type IntoIter = <&'a SVector<usize, D> as IntoIterator>::IntoIter;
    type Item = &'a usize;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a mut LatticePoint<D> {
    type IntoIter = <&'a mut SVector<usize, D> as IntoIterator>::IntoIter;
    type Item = &'a mut usize;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<const D: usize> Index<usize> for LatticePoint<D> {
    type Output = usize;

    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// let point = LatticePoint::new([0; 4].into());
    /// let _ = point[4];
    /// ```
    #[inline]
    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl<const D: usize> IndexMut<usize> for LatticePoint<D> {
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// let mut point = LatticePoint::new([0; 4].into());
    /// point[4] += 1;
    /// ```
    #[inline]
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl<T, const D: usize> From<T> for LatticePoint<D>
where
    SVector<usize, D>: From<T>,
{
    #[inline]
    fn from(data: T) -> Self {
        Self::new(SVector::from(data))
    }
}

impl<const D: usize> From<LatticePoint<D>> for [usize; D]
where
    SVector<usize, D>: Into<[usize; D]>,
{
    #[inline]
    fn from(data: LatticePoint<D>) -> [usize; D] {
        data.data.into()
    }
}

impl<const D: usize> AsRef<SVector<usize, D>> for LatticePoint<D> {
    #[inline]
    fn as_ref(&self) -> &SVector<usize, D> {
        self.as_svector()
    }
}

impl<const D: usize> AsMut<SVector<usize, D>> for LatticePoint<D> {
    #[inline]
    fn as_mut(&mut self) -> &mut SVector<usize, D> {
        self.as_svector_mut()
    }
}

impl<const D: usize> NumberOfLatticeElement<D> for LatticePoint<D> {
    #[inline]
    fn number_of_elements(lattice: &LatticeCyclic<D>) -> usize {
        lattice.number_of_points()
    }
}

impl<const D: usize> IndexToElement<D> for LatticePoint<D> {
    // TODO
    #[inline]
    fn index_to_element(lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
        Self::index_to_point(lattice, index)
    }
}

/// Trait to convert an element on a lattice to an [`usize`].
///
/// Used mainly to index field on the lattice using [`std::vec::Vec`]
// TODO change name ? make it less confusing
pub trait LatticeElementToIndex<const D: usize> {
    /// Given a lattice return an index from the element
    #[must_use]
    fn to_index(&self, l: &LatticeCyclic<D>) -> usize;
}

impl<const D: usize> LatticeElementToIndex<D> for LatticePoint<D> {
    #[inline]
    fn to_index(&self, l: &LatticeCyclic<D>) -> usize {
        self.iter()
            .enumerate()
            .map(|(index, pos)| {
                (pos % l.dim()) * l.dim().pow(index.try_into().expect("conversion error"))
            })
            .sum()
    }
}

impl<const D: usize> LatticeElementToIndex<D> for Direction<D> {
    /// equivalent to [`Direction::to_index()`]
    #[inline]
    fn to_index(&self, _: &LatticeCyclic<D>) -> usize {
        self.index()
    }
}

impl<const D: usize> LatticeElementToIndex<D> for LatticeLinkCanonical<D> {
    #[inline]
    fn to_index(&self, l: &LatticeCyclic<D>) -> usize {
        self.pos().to_index(l) * D + self.dir().index()
    }
}

/// This is just the identity.
///
/// It is implemented for compatibility reason
/// such that function that require a [`LatticeElementToIndex`] can also accept [`usize`].
impl<const D: usize> LatticeElementToIndex<D> for usize {
    /// return self
    #[inline]
    fn to_index(&self, _l: &LatticeCyclic<D>) -> usize {
        *self
    }
}

/// A trait to get the number of certain type of item there is on the lattice.
pub trait NumberOfLatticeElement<const D: usize>: Sealed {
    /// Returns the number of items there is on the lattice.
    #[must_use]
    fn number_of_elements(lattice: &LatticeCyclic<D>) -> usize;
}

/// A trait to convert an index to a element on a [`LatticeCyclic`].
pub trait IndexToElement<const D: usize>: Sealed + LatticeElementToIndex<D> + Sized {
    /// Converts an index into an element.
    #[must_use]
    fn index_to_element(lattice: &LatticeCyclic<D>, index: usize) -> Option<Self>;
}

// #[doc(hidden)]
impl Sealed for () {}

impl<const D: usize> LatticeElementToIndex<D> for () {
    #[inline]
    fn to_index(&self, _l: &LatticeCyclic<D>) -> usize {
        0
    }
}

impl<const D: usize> NumberOfLatticeElement<D> for () {
    #[inline]
    fn number_of_elements(_lattice: &LatticeCyclic<D>) -> usize {
        1
    }
}

impl<const D: usize> IndexToElement<D> for () {
    #[inline]
    fn index_to_element(_lattice: &LatticeCyclic<D>, index: usize) -> Option<()> {
        (index == 0).then_some(())
    }
}

/// A canonical link of a lattice. It contain a position and a direction.
///
/// The direction should always be positive.
/// By itself the link does not store data about the lattice. Hence most function require a [`LatticeCyclic`].
/// It also means that there is no guarantee that the object is inside a lattice.
/// You can use modulus over the elements to use inside a lattice.
///
/// This object can be used to safely index in a [`std::collections::HashMap`]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy, Sealed)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeLinkCanonical<const D: usize> {
    from: LatticePoint<D>,
    dir: Direction<D>,
}

impl<const D: usize> LatticeLinkCanonical<D> {
    /// Try create a [`LatticeLinkCanonical`]. If the dir is negative it fails.
    ///
    /// To guaranty creating an element see [`LatticeCyclic::link_canonical`].
    /// The creation of an element this ways does not guaranties that the element is inside a lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// let l = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XNeg.into());
    /// assert_eq!(l, None);
    ///
    /// let l = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XPos.into());
    /// assert!(l.is_some());
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(from: LatticePoint<D>, dir: Direction<D>) -> Option<Self> {
        if dir.is_negative() {
            return None;
        }
        Some(Self { from, dir })
    }

    /// Position of the link.
    #[must_use]
    #[inline]
    pub const fn pos(&self) -> &LatticePoint<D> {
        &self.from
    }

    /// Get a mutable reference on the position of the link.
    #[must_use]
    #[inline]
    pub fn pos_mut(&mut self) -> &mut LatticePoint<D> {
        &mut self.from
    }

    /// Direction of the link.
    #[must_use]
    #[inline]
    pub const fn dir(&self) -> &Direction<D> {
        &self.dir
    }

    /// Set the direction to dir
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut lattice_link_canonical =
    ///     LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::YPos.into())
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// lattice_link_canonical.set_dir(DirectionEnum::XPos.into());
    /// assert_eq!(*lattice_link_canonical.dir(), DirectionEnum::XPos.into());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    /// panic if a negative direction is given.
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut lattice_link_canonical = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// lattice_link_canonical.set_dir(DirectionEnum::XNeg.into()); // Panics !
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_dir(&mut self, dir: Direction<D>) {
        assert!(
            !dir.is_negative(),
            "Cannot set a negative direction to a canonical link."
        );
        self.dir = dir;
    }

    /// Set the direction using positive direction. i.e. if a direction `-x` is passed
    /// the direction assigned will be `+x`.
    ///
    /// This is equivalent to `link.set_dir(dir.to_positive())`.
    #[inline]
    pub fn set_dir_positive(&mut self, dir: Direction<D>) {
        self.dir = dir.to_positive();
    }

    #[inline]
    #[must_use]
    fn index_to_canonical_link(lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
        Self::new(
            LatticePoint::index_to_element(lattice, index / D)?,
            Direction::new(index % D, true)?,
        )
    }
}

impl<const D: usize> Display for LatticeLinkCanonical<D> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "canonical link [position {}, direction {}]",
            self.from, self.dir
        )
    }
}

impl<const D: usize> From<LatticeLinkCanonical<D>> for LatticeLink<D> {
    #[inline]
    fn from(l: LatticeLinkCanonical<D>) -> Self {
        Self::new(l.from, l.dir)
    }
}

impl<const D: usize> From<&LatticeLinkCanonical<D>> for LatticeLink<D> {
    #[inline]
    fn from(l: &LatticeLinkCanonical<D>) -> Self {
        Self::new(l.from, l.dir)
    }
}

impl<const D: usize> NumberOfLatticeElement<D> for LatticeLinkCanonical<D> {
    #[inline]
    fn number_of_elements(lattice: &LatticeCyclic<D>) -> usize {
        lattice.number_of_canonical_links_space()
    }
}

impl<const D: usize> IndexToElement<D> for LatticeLinkCanonical<D> {
    // TODO
    #[inline]
    fn index_to_element(lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
        Self::index_to_canonical_link(lattice, index)
    }
}

/// A lattice link, contrary to [`LatticeLinkCanonical`] the direction can be negative.
///
/// This means that multiple link can be equivalent but does not have the same data
/// and therefore hash (hopefully).
///
/// By itself the link does not store data about the lattice. Hence most function require a [`LatticeCyclic`].
/// It also means that there is no guarantee that the object is inside a lattice.
/// You can use modulus over the elements to use inside a lattice.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeLink<const D: usize> {
    from: LatticePoint<D>,
    dir: Direction<D>,
}

impl<const D: usize> LatticeLink<D> {
    /// Create a link from position `from` and direction `dir`.
    #[must_use]
    #[inline]
    pub const fn new(from: LatticePoint<D>, dir: Direction<D>) -> Self {
        Self { from, dir }
    }

    /// Get the position of the link.
    #[must_use]
    #[inline]
    pub const fn pos(&self) -> &LatticePoint<D> {
        &self.from
    }

    /// Get a mutable reference to the position of the link.
    #[must_use]
    #[inline]
    pub fn pos_mut(&mut self) -> &mut LatticePoint<D> {
        &mut self.from
    }

    /// Get the direction of the link.
    #[must_use]
    #[inline]
    pub const fn dir(&self) -> &Direction<D> {
        &self.dir
    }

    /// Get a mutable reference to the direction of the link.
    #[must_use]
    #[inline]
    pub fn dir_mut(&mut self) -> &mut Direction<D> {
        &mut self.dir
    }

    /// Get if the direction of the link is positive.
    #[must_use]
    #[inline]
    pub const fn is_dir_positive(&self) -> bool {
        self.dir.is_positive()
    }

    /// Get if the direction of the link is negative.
    #[must_use]
    #[inline]
    pub const fn is_dir_negative(&self) -> bool {
        self.dir.is_negative()
    }
}

impl<const D: usize> Display for LatticeLink<D> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "link [position {}, direction {}]", self.from, self.dir)
    }
}

/// Represent a cardinal direction
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy, Sealed)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Direction<const D: usize> {
    index_dir: usize,
    is_positive: bool,
}

impl<const D: usize> Direction<D> {
    /// New direction from a direction as an idex and wether it is in the positive direction.
    #[must_use]
    #[inline]
    pub const fn new(index_dir: usize, is_positive: bool) -> Option<Self> {
        // TODO return error ?
        if index_dir >= D {
            return None;
        }
        Some(Self {
            index_dir,
            is_positive,
        })
    }

    /// List of all positives directions.
    /// This is very slow use [`Self::positive_directions`] instead.
    #[allow(clippy::missing_panics_doc)] // it nevers panic
    #[must_use]
    #[inline]
    pub fn positives_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(D);
        for i in 0..D {
            x.push(Self::new(i, true).expect("unreachable"));
        }
        x
    }

    /// List all directions.
    /// This is very slow use [`DirectionList::directions`] instead.
    #[allow(clippy::missing_panics_doc)] // it nevers panic
    #[must_use]
    #[inline]
    pub fn directions_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(2 * D);
        for i in 0..D {
            x.push(Self::new(i, true).expect("unreachable"));
            x.push(Self::new(i, false).expect("unreachable"));
        }
        x
    }

    // TODO add const function for all direction once operation on const generic are added
    /// Get all direction with the sign `IS_POSITIVE`.
    #[must_use]
    #[inline]
    pub const fn directions_array<const IS_POSITIVE: bool>() -> [Self; D] {
        // TODO use unsafe code to avoid the allocation
        let mut i = 0_usize;
        let mut array = [Self {
            index_dir: 0,
            is_positive: IS_POSITIVE,
        }; D];
        while i < D {
            array[i] = Self {
                index_dir: i,
                is_positive: IS_POSITIVE,
            };
            i += 1;
        }
        array
    }

    /// Get all negative direction
    #[must_use]
    #[inline]
    pub const fn negative_directions() -> [Self; D] {
        Self::directions_array::<false>()
    }

    /// Get all positive direction
    #[allow(clippy::same_name_method)]
    #[must_use]
    #[inline]
    pub const fn positive_directions() -> [Self; D] {
        Self::directions_array::<true>()
    }

    /// Get if the position is positive.
    #[must_use]
    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.is_positive
    }

    /// Get if the position is Negative. see [`Direction::is_positive`]
    #[must_use]
    #[inline]
    pub const fn is_negative(&self) -> bool {
        !self.is_positive()
    }

    /// Return the positive direction associated, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// assert_eq!(
    ///     Direction::<4>::new(1, false).unwrap().to_positive(),
    ///     Direction::<4>::new(1, true).unwrap()
    /// );
    /// assert_eq!(
    ///     Direction::<4>::new(1, true).unwrap().to_positive(),
    ///     Direction::<4>::new(1, true).unwrap()
    /// );
    /// ```
    #[must_use]
    #[inline]
    pub const fn to_positive(mut self) -> Self {
        self.is_positive = true;
        self
    }

    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{lattice::Direction, error::ImplementationError};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert_eq!(
    ///     Direction::<4>::new(1, false)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     1
    /// );
    /// assert_eq!(
    ///     Direction::<4>::new(3, true)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     3
    /// );
    /// assert_eq!(
    ///     Direction::<6>::new(5, false)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     5
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn index(&self) -> usize {
        self.index_dir
    }

    /// Convert the direction into a vector of norm `a`;
    #[must_use]
    #[inline]
    pub fn to_vector(self, a: f64) -> SVector<Real, D> {
        self.to_unit_vector() * a
    }

    /// Returns the dimension
    #[must_use]
    #[inline]
    pub const fn dim() -> usize {
        D
    }

    /// Convert the direction into a vector of norm `1`;
    #[must_use]
    #[inline]
    pub fn to_unit_vector(self) -> SVector<Real, D> {
        let mut v = SVector::zeros();
        v[self.index_dir] = 1_f64;
        v
    }

    /// Find the direction the vector point the most.
    /// For a zero vector return [`DirectionEnum::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box< dyn std::error::Error>> {
    /// assert_eq!(
    ///     Direction::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)),
    ///     Direction::<4>::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     Direction::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)),
    ///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     Direction::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)),
    ///     Direction::<4>::new(3, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::missing_panics_doc)] // it nevers panic
    #[must_use]
    #[inline]
    pub fn from_vector(v: &SVector<Real, D>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for (i, dir) in Self::positive_directions().iter().enumerate() {
            let scalar_prod = v.dot(&dir.to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        Self::new(index_max, is_positive).expect("Unreachable")
    }
}

// TODO default when condition on const generic is available

/*
impl<const D: usize> Default for LatticeLinkCanonical<D>
where
    Direction<D>: Default,
{
    fn default() -> Self {
        Self {
            from: LatticePoint::default(),
            dir: Direction::default(),
        }
    }
}

impl<const D: usize> Default for LatticeLink<D>
where
    Direction<D>: Default,
{
    fn default() -> Self {
        Self {
            from: LatticePoint::default(),
            dir: Direction::default(),
        }
    }
}
*/

impl<const D: usize> Display for Direction<D> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[index {}, is positive {}]",
            self.index(),
            self.is_positive()
        )
    }
}

// /// TODO impl doc
// impl<const D: usize> NumberOfLatticeElement<D> for Direction<D> {
//     #[inline]
//     fn number_of_elements(_lattice: &LatticeCyclic<D>) -> usize {
//         D
//     }
// }

// // TODO impl doc
// impl<const D: usize> IndexToElement<D> for Direction<D> {
//     fn index_to_element(_lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
//         Self::new(index, true)
//     }
// }

/// List all possible direction
pub trait DirectionList: Sized {
    /// List all directions.
    #[must_use]
    fn directions() -> &'static [Self];
    /// List all positive directions.
    #[must_use]
    fn positive_directions() -> &'static [Self];
}

/// Error return by [`TryFrom`] for [`Direction`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum DirectionConversionError {
    /// The index is out of bound, i.e. the direction axis does not exist in the lower space dimension.
    IndexOutOfBound, // more error info like dim and index
}

impl Display for DirectionConversionError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBound => write!(f, "the index is out of bound, the direction axis does not exist in the lower space dimension"),
        }
    }
}

impl Error for DirectionConversionError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IndexOutOfBound => None,
        }
    }
}

implement_direction_list!();

implement_direction_from!();

/// Partial ordering is set as follows: two directions can be compared if they have the same index
/// or the same direction sign. In the first case a positive direction is greater than a negative direction
/// In the latter case the ordering is done on the index.
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::Direction;
/// # use lattice_qcd_rs::error::ImplementationError;
/// use std::cmp::Ordering;
/// # fn main() -> Result<(), Box< dyn std::error::Error>> {
///
/// let dir_1 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_2 =
///     Direction::<4>::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert!(dir_1 < dir_2);
/// assert_eq!(dir_1.partial_cmp(&dir_2), Some(Ordering::Less));
/// //--------
/// let dir_3 =
///     Direction::<4>::new(3, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_4 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert!(dir_3 > dir_4);
/// assert_eq!(dir_3.partial_cmp(&dir_4), Some(Ordering::Greater));
/// //--------
/// let dir_5 =
///     Direction::<4>::new(3, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_6 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert_eq!(dir_5.partial_cmp(&dir_6), None);
/// //--------
/// let dir_5 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_6 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert_eq!(dir_5.partial_cmp(&dir_6), Some(Ordering::Equal));
/// # Ok(())
/// # }
/// ```
impl<const D: usize> PartialOrd for Direction<D> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.is_positive() == other.is_positive() {
            self.index().partial_cmp(&other.index())
        } else if self.index() == other.index() {
            self.is_positive().partial_cmp(&other.is_positive())
        } else {
            None
        }
    }
}

impl<const D: usize> Neg for Direction<D> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.is_positive = !self.is_positive;
        self
    }
}

impl<const D: usize> Neg for &Direction<D> {
    type Output = Direction<D>;

    #[inline]
    fn neg(self) -> Self::Output {
        -*self
    }
}

/// Return [`Direction::to_index`].
impl<const D: usize> From<Direction<D>> for usize {
    #[inline]
    fn from(d: Direction<D>) -> Self {
        d.index()
    }
}

/// Return [`Direction::to_index`].
impl<const D: usize> From<&Direction<D>> for usize {
    #[inline]
    fn from(d: &Direction<D>) -> Self {
        d.index()
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<const D: usize> From<SVector<Real, D>> for Direction<D> {
    #[inline]
    fn from(v: SVector<Real, D>) -> Self {
        Self::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<const D: usize> From<&SVector<Real, D>> for Direction<D> {
    #[inline]
    fn from(v: &SVector<Real, D>) -> Self {
        Self::from_vector(v)
    }
}

/// Return [`Direction::to_unit_vector`].
impl<const D: usize> From<Direction<D>> for SVector<Real, D> {
    #[inline]
    fn from(d: Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`Direction::to_unit_vector`].
impl<const D: usize> From<&Direction<D>> for SVector<Real, D> {
    #[inline]
    fn from(d: &Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

impl From<DirectionEnum> for Direction<4> {
    #[inline]
    fn from(d: DirectionEnum) -> Self {
        Self::new(DirectionEnum::index(d), d.is_positive()).expect("unreachable")
    }
}

impl From<Direction<4>> for DirectionEnum {
    #[inline]
    fn from(d: Direction<4>) -> Self {
        d.into()
    }
}

impl From<&Direction<4>> for DirectionEnum {
    #[inline]
    fn from(d: &Direction<4>) -> Self {
        match (d.index(), d.is_positive()) {
            (0, true) => Self::XPos,
            (0, false) => Self::XNeg,
            (1, true) => Self::YPos,
            (1, false) => Self::YNeg,
            (2, true) => Self::ZPos,
            (2, false) => Self::ZNeg,
            (3, true) => Self::TPos,
            (3, false) => Self::TNeg,
            _ => unreachable!(),
        }
    }
}

impl From<&DirectionEnum> for Direction<4> {
    #[inline]
    fn from(d: &DirectionEnum) -> Self {
        Self::new(DirectionEnum::index(*d), d.is_positive()).expect("unreachable")
    }
}

// TODO depreciate ?
/// Represent a cardinal direction
#[allow(clippy::exhaustive_enums)]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum DirectionEnum {
    /// Positive x direction.
    XPos,
    /// Negative x direction.
    XNeg,
    /// Positive y direction.
    YPos,
    /// Negative y direction.
    YNeg,
    /// Positive z direction.
    ZPos,
    /// Negative z direction.
    ZNeg,
    /// Positive t direction.
    TPos,
    /// Negative t direction.
    TNeg,
}

impl DirectionEnum {
    /// List all directions.
    pub const DIRECTIONS: [Self; 8] = [
        Self::XPos,
        Self::YPos,
        Self::ZPos,
        Self::TPos,
        Self::XNeg,
        Self::YNeg,
        Self::ZNeg,
        Self::TNeg,
    ];
    /// List all spatial directions.
    pub const DIRECTIONS_SPACE: [Self; 6] = [
        Self::XPos,
        Self::YPos,
        Self::ZPos,
        Self::XNeg,
        Self::YNeg,
        Self::ZNeg,
    ];
    /// List of all positives directions.
    pub const POSITIVES: [Self; 4] = [Self::XPos, Self::YPos, Self::ZPos, Self::TPos];
    /// List spatial positive direction.
    pub const POSITIVES_SPACE: [Self; 3] = [Self::XPos, Self::YPos, Self::ZPos];

    /// Convert the direction into a vector of norm `a`;
    #[must_use]
    #[inline]
    pub fn to_vector(self, a: f64) -> Vector4<Real> {
        self.to_unit_vector() * a
    }

    /// Convert the direction into a vector of norm `1`;
    #[must_use]
    #[inline]
    pub const fn to_unit_vector(self) -> Vector4<Real> {
        match self {
            Self::XPos => Vector4::<Real>::new(1_f64, 0_f64, 0_f64, 0_f64),
            Self::XNeg => Vector4::<Real>::new(-1_f64, 0_f64, 0_f64, 0_f64),
            Self::YPos => Vector4::<Real>::new(0_f64, 1_f64, 0_f64, 0_f64),
            Self::YNeg => Vector4::<Real>::new(0_f64, -1_f64, 0_f64, 0_f64),
            Self::ZPos => Vector4::<Real>::new(0_f64, 0_f64, 1_f64, 0_f64),
            Self::ZNeg => Vector4::<Real>::new(0_f64, 0_f64, -1_f64, 0_f64),
            Self::TPos => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, 1_f64),
            Self::TNeg => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, -1_f64),
        }
    }

    /// Get if the position is positive.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XPos.is_positive(), true);
    /// assert_eq!(DirectionEnum::TPos.is_positive(), true);
    /// assert_eq!(DirectionEnum::YNeg.is_positive(), false);
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_positive(self) -> bool {
        match self {
            Self::XPos | Self::YPos | Self::ZPos | Self::TPos => true,
            Self::XNeg | Self::YNeg | Self::ZNeg | Self::TNeg => false,
        }
    }

    /// Get if the position is Negative. see [`DirectionEnum::is_positive`]
    #[must_use]
    #[inline]
    pub const fn is_negative(self) -> bool {
        !self.is_positive()
    }

    /// Find the direction the vector point the most.
    /// For a zero vector return [`DirectionEnum::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// # extern crate nalgebra;
    /// assert_eq!(
    ///     DirectionEnum::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)),
    ///     DirectionEnum::XPos
    /// );
    /// assert_eq!(
    ///     DirectionEnum::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)),
    ///     DirectionEnum::YNeg
    /// );
    /// assert_eq!(
    ///     DirectionEnum::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)),
    ///     DirectionEnum::TPos
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_vector(v: &Vector4<Real>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for i in 0..Self::POSITIVES.len() {
            let scalar_prod = v.dot(&Self::POSITIVES[i].to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        match index_max {
            0 => {
                if is_positive {
                    Self::XPos
                } else {
                    Self::XNeg
                }
            }
            1 => {
                if is_positive {
                    Self::YPos
                } else {
                    Self::YNeg
                }
            }
            2 => {
                if is_positive {
                    Self::ZPos
                } else {
                    Self::ZNeg
                }
            }
            3 => {
                if is_positive {
                    Self::TPos
                } else {
                    Self::TNeg
                }
            }
            _ => {
                // the code should attain this code. and therefore panicking is not expected.
                unreachable!("Implementation error : invalid index")
            }
        }
    }

    /// Return the positive direction associated, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XNeg.to_positive(), DirectionEnum::XPos);
    /// assert_eq!(DirectionEnum::YPos.to_positive(), DirectionEnum::YPos);
    /// ```
    #[inline]
    #[must_use]
    pub const fn to_positive(self) -> Self {
        match self {
            Self::XNeg => Self::XPos,
            Self::YNeg => Self::YPos,
            Self::ZNeg => Self::ZPos,
            Self::TNeg => Self::TPos,
            _ => self,
        }
    }

    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XPos.index(), 0);
    /// assert_eq!(DirectionEnum::XNeg.index(), 0);
    /// assert_eq!(DirectionEnum::YPos.index(), 1);
    /// assert_eq!(DirectionEnum::YNeg.index(), 1);
    /// assert_eq!(DirectionEnum::ZPos.index(), 2);
    /// assert_eq!(DirectionEnum::ZNeg.index(), 2);
    /// assert_eq!(DirectionEnum::TPos.index(), 3);
    /// assert_eq!(DirectionEnum::TNeg.index(), 3);
    /// ```
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::XPos | Self::XNeg => 0,
            Self::YPos | Self::YNeg => 1,
            Self::ZPos | Self::ZNeg => 2,
            Self::TPos | Self::TNeg => 3,
        }
    }
}

/// Return [`DirectionEnum::XPos`]
impl Default for DirectionEnum {
    ///Return [`DirectionEnum::XPos`]
    #[inline]
    fn default() -> Self {
        Self::XPos
    }
}

impl Display for DirectionEnum {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::XPos => write!(f, "positive X direction"),
            Self::XNeg => write!(f, "negative X direction"),
            Self::YPos => write!(f, "positive Y direction"),
            Self::YNeg => write!(f, "negative Y direction"),
            Self::ZPos => write!(f, "positive Z direction"),
            Self::ZNeg => write!(f, "negative Z direction"),
            Self::TPos => write!(f, "positive T direction"),
            Self::TNeg => write!(f, "negative T direction"),
        }
    }
}

impl DirectionList for DirectionEnum {
    #[inline]
    fn directions() -> &'static [Self] {
        &Self::DIRECTIONS
    }

    #[inline]
    fn positive_directions() -> &'static [Self] {
        &Self::POSITIVES
    }
}

impl PartialOrd for DirectionEnum {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Direction::<4>::from(self).partial_cmp(&other.into())
    }
}

/// Return the negative of a direction
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::DirectionEnum;
/// assert_eq!(-DirectionEnum::XNeg, DirectionEnum::XPos);
/// assert_eq!(-DirectionEnum::YPos, DirectionEnum::YNeg);
/// ```
impl Neg for DirectionEnum {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Self::XPos => Self::XNeg,
            Self::XNeg => Self::XPos,
            Self::YPos => Self::YNeg,
            Self::YNeg => Self::YPos,
            Self::ZPos => Self::ZNeg,
            Self::ZNeg => Self::ZPos,
            Self::TPos => Self::TNeg,
            Self::TNeg => Self::TPos,
        }
    }
}

impl Neg for &DirectionEnum {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            DirectionEnum::XPos => &DirectionEnum::XNeg,
            DirectionEnum::XNeg => &DirectionEnum::XPos,
            DirectionEnum::YPos => &DirectionEnum::YNeg,
            DirectionEnum::YNeg => &DirectionEnum::YPos,
            DirectionEnum::ZPos => &DirectionEnum::ZNeg,
            DirectionEnum::ZNeg => &DirectionEnum::ZPos,
            DirectionEnum::TPos => &DirectionEnum::TNeg,
            DirectionEnum::TNeg => &DirectionEnum::TPos,
        }
    }
}

/// Return [`DirectionEnum::to_index`].
impl From<DirectionEnum> for usize {
    #[inline]
    fn from(d: DirectionEnum) -> Self {
        d.index()
    }
}

/// Return [`DirectionEnum::to_index`].
impl From<&DirectionEnum> for usize {
    #[inline]
    fn from(d: &DirectionEnum) -> Self {
        DirectionEnum::index(*d)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<Vector4<Real>> for DirectionEnum {
    #[inline]
    fn from(v: Vector4<Real>) -> Self {
        Self::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<&Vector4<Real>> for DirectionEnum {
    #[inline]
    fn from(v: &Vector4<Real>) -> Self {
        Self::from_vector(v)
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<DirectionEnum> for Vector4<Real> {
    #[inline]
    fn from(d: DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<&DirectionEnum> for Vector4<Real> {
    #[inline]
    fn from(d: &DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

impl LatticeElementToIndex<4> for DirectionEnum {
    #[inline]
    fn to_index(&self, l: &LatticeCyclic<4>) -> usize {
        Direction::<4>::from(self).to_index(l)
    }
}

// impl Sealed for DirectionEnum {}

// impl NumberOfLatticeElement<4> for DirectionEnum {
//     #[inline]
//     fn number_of_elements(lattice: &LatticeCyclic<4>) -> usize {
//         Direction::<4>::number_of_elements(lattice)
//     }
// }

// impl IndexToElement<4> for DirectionEnum {
//     fn index_to_element(lattice: &LatticeCyclic<4>, index: usize) -> Option<Self> {
//         Direction::<4>::index_to_element(lattice, index).map(Into::into)
//     }
// }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn directions_list_cst() {
        let dirs = Direction::<3>::positive_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(Some(*dir), Direction::new(index, true));
        }
        let dirs = Direction::<3>::negative_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(Some(*dir), Direction::new(index, false));
        }

        let dirs = Direction::<8>::positive_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(Some(*dir), Direction::new(index, true));
        }
        let dirs = Direction::<8>::negative_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(Some(*dir), Direction::new(index, false));
        }

        let dirs = Direction::<0>::positive_directions();
        assert_eq!(dirs.len(), 0);
        let dirs = Direction::<0>::negative_directions();
        assert_eq!(dirs.len(), 0);

        let dirs = Direction::<128>::positive_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(Some(*dir), Direction::new(index, true));
        }
        let dirs = Direction::<128>::negative_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(Some(*dir), Direction::new(index, false));
        }

        let dirs = Direction::<128>::positive_directions();
        let dir_vec = Direction::<128>::positives_vec();
        assert_eq!(dirs.to_vec(), dir_vec);

        let dir_vec = Direction::<128>::directions_vec();
        assert_eq!(dir_vec.len(), 256);
    }

    #[test]
    #[allow(clippy::explicit_counter_loop)]
    fn lattice_pt() {
        let array = [1, 2, 3, 4];
        let mut pt = LatticePoint::from(array);
        assert_eq!(LatticePoint::from(array), LatticePoint::new(array.into()));

        assert_eq!(<[usize; 4]>::from(pt), array);
        assert!(!pt.is_empty());
        assert_eq!(pt.iter().count(), 4);
        assert_eq!(pt.iter_mut().count(), 4);
        assert_eq!(pt.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(pt.iter_mut().collect::<Vec<_>>(), vec![&1, &2, &3, &4]);

        let mut j = 0;
        for i in &pt {
            assert_eq!(*i, array[j]);
            j += 1;
        }

        let mut j = 0;
        for i in &mut pt {
            assert_eq!(*i, array[j]);
            j += 1;
        }

        pt.iter_mut().for_each(|el| *el = 0);
        assert_eq!(pt, LatticePoint::new_zero());
        assert_eq!(LatticePoint::<4>::default(), LatticePoint::new_zero());

        let mut pt_2 = LatticePoint::<0>::new_zero();
        assert!(pt_2.is_empty());
        assert_eq!(pt_2.iter().count(), 0);
        assert_eq!(pt_2.iter_mut().count(), 0);

        assert_eq!(pt.to_string(), pt.as_ref().to_string());
    }

    #[test]
    #[should_panic(expected = "Cannot set a negative direction to a canonical link.")]
    fn set_dir_neg() {
        let mut lattice_link_canonical =
            LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XPos.into())
                .expect("it is made to exist");

        lattice_link_canonical.set_dir(DirectionEnum::XNeg.into());
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn dir() {
        assert!(Direction::<4>::new(4, true).is_none());
        assert!(Direction::<4>::new(32, true).is_none());
        assert!(Direction::<4>::new(3, true).is_some());
        assert!(Direction::<127>::new(128, true).is_none());
        assert_eq!(Direction::<4>::dim(), 4);
        assert_eq!(Direction::<124>::dim(), 124);

        assert!(DirectionEnum::TNeg.is_negative());
        assert!(!DirectionEnum::ZPos.is_negative());

        assert_eq!(-&DirectionEnum::TNeg, &DirectionEnum::TPos);
        assert_eq!(-&DirectionEnum::TPos, &DirectionEnum::TNeg);
        assert_eq!(-&DirectionEnum::ZNeg, &DirectionEnum::ZPos);
        assert_eq!(-&DirectionEnum::ZPos, &DirectionEnum::ZNeg);
        assert_eq!(-&DirectionEnum::YNeg, &DirectionEnum::YPos);
        assert_eq!(-&DirectionEnum::YPos, &DirectionEnum::YNeg);
        assert_eq!(-&DirectionEnum::XNeg, &DirectionEnum::XPos);
        assert_eq!(-&DirectionEnum::XPos, &DirectionEnum::XNeg);

        assert_eq!(-DirectionEnum::TNeg, DirectionEnum::TPos);
        assert_eq!(-DirectionEnum::TPos, DirectionEnum::TNeg);
        assert_eq!(-DirectionEnum::ZNeg, DirectionEnum::ZPos);
        assert_eq!(-DirectionEnum::ZPos, DirectionEnum::ZNeg);
        assert_eq!(-DirectionEnum::YNeg, DirectionEnum::YPos);
        assert_eq!(-DirectionEnum::YPos, DirectionEnum::YNeg);
        assert_eq!(-DirectionEnum::XNeg, DirectionEnum::XPos);
        assert_eq!(-DirectionEnum::XPos, DirectionEnum::XNeg);

        assert_eq!(DirectionEnum::directions().len(), 8);
        assert_eq!(DirectionEnum::positive_directions().len(), 4);

        assert_eq!(Direction::<4>::directions().len(), 8);
        assert_eq!(Direction::<4>::positive_directions().len(), 4);

        let l = LatticeCyclic::new(1_f64, 4).expect("lattice has an error");
        for dir in Direction::<3>::directions() {
            assert_eq!(
                <Direction<3> as LatticeElementToIndex<3>>::to_index(dir, &l),
                dir.index()
            );
        }

        let array_dir_name = ["X", "Y", "Z", "T"];
        let array_pos = ["positive", "negative"];
        for (i, dir) in DirectionEnum::directions().iter().enumerate() {
            assert_eq!(
                dir.to_string(),
                format!("{} {} direction", array_pos[i / 4], array_dir_name[i % 4])
            );
        }
    }

    /// In this test we test the trait [`From`] and [`TryFrom`] implemented automatically by
    /// [`implement_direction_from`].
    #[test]
    fn direction_conversion() -> Result<(), DirectionConversionError> {
        // try into test
        let dir = Direction::<4>::new(2, true).ok_or(DirectionConversionError::IndexOutOfBound)?;
        assert_eq!(
            (&dir).try_into(),
            Direction::<3>::new(2, true).ok_or(DirectionConversionError::IndexOutOfBound)
        );

        // failing try into test
        let dir = Direction::<4>::new(3, true).ok_or(DirectionConversionError::IndexOutOfBound)?;
        assert_eq!(
            (&dir).try_into(),
            Result::<Direction<3>, DirectionConversionError>::Err(
                DirectionConversionError::IndexOutOfBound,
            )
        );

        // into test
        let dir = Direction::<3>::new(2, true).ok_or(DirectionConversionError::IndexOutOfBound)?;
        assert_eq!(
            <&Direction<3> as Into<Direction::<4>>>::into(
                #[allow(clippy::needless_borrows_for_generic_args)] // false positive
                &dir
            ),
            Direction::<4>::new(2, true).ok_or(DirectionConversionError::IndexOutOfBound)?
        );

        Ok(())
    }

    #[test]
    fn lattice_link() {
        let pt = [0, 0, 0, 0].into();
        let dir = DirectionEnum::XNeg.into();
        let mut link =
            LatticeLinkCanonical::new(pt, DirectionEnum::YPos.into()).expect("link has an error");
        link.set_dir_positive(dir);
        assert_eq!(Some(link), LatticeLinkCanonical::new(pt, dir.to_positive()));

        let mut link = LatticeLink::new(pt, DirectionEnum::YPos.into());
        let pos = *link.pos();
        let dir = *link.dir();
        assert_eq!(pos, *link.pos_mut());
        assert_eq!(dir, *link.dir_mut());
        assert!(link.is_dir_positive());
        *link.dir_mut() = DirectionEnum::YNeg.into();
        assert!(!link.is_dir_positive());

        assert_eq!(
            link.to_string(),
            format!(
                "link [position {}, direction [index 1, is positive false]]",
                SVector::<usize, 4>::zeros()
            )
        );

        let vector = SVector::<usize, 5>::from([1, 0, 0, 0, 0]);
        let canonical_link = LatticeLinkCanonical::<5>::new(
            LatticePoint::new(vector),
            Direction::new(0, true).expect("direction cannot be constructed"),
        )
        .expect("link cannot be constructed");
        println!("{canonical_link}");
        assert_eq!(
            canonical_link.to_string(),
            format!("canonical link [position {vector}, direction [index 0, is positive true]]")
        );
    }
}
