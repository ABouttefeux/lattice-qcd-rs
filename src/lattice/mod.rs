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

mod direction;
mod iterator;
mod lattice_cyclic;

use std::convert::TryInto;
use std::fmt::{self, Display};
use std::iter::FusedIterator;
use std::ops::{Index, IndexMut};

use nalgebra::SVector;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::Sealed;

pub use self::direction::{
    Axis, Direction, DirectionConversionError, DirectionEnum, DirectionList, OrientedDirection,
};
// TODO remove IteratorElement from public interface ?
pub use self::iterator::{
    IteratorDirection, IteratorElement, IteratorLatticeLinkCanonical, IteratorLatticePoint,
    LatticeIterator, ParIter, ParIterLatticeLinkCanonical, ParIterLatticePoint,
};
pub use self::lattice_cyclic::LatticeCyclic;
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

    /// transform an index to a point inside a given lattice if it exists
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
    /// The starting point of the link.
    from: LatticePoint<D>,
    /// The direction of the link.
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
    /// The starting point of the link.
    from: LatticePoint<D>,
    /// The direction of the link.
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
        for (index, dir) in Direction::<3>::directions().iter().enumerate() {
            assert_eq!(
                <Direction<3> as LatticeElementToIndex<3>>::to_index(dir, &l),
                index
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
