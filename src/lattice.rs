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

use std::cmp::Ordering;
use std::convert::TryInto;
use std::iter::FusedIterator;
use std::ops::{Index, IndexMut, Neg};

use lattice_qcd_rs_procedural_macro::{implement_direction_from, implement_direction_list};
use na::{SVector, Vector4};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::field::{EField, LinkMatrix};
use super::{error::LatticeInitializationError, Real};

/// A Cyclic lattice in space. Does not store point and links but is used to generate them.
///
/// The generic parameter `D` is the dimension.
///
/// This lattice is Cyclic more precisely if the lattice has N points in each direction.
/// Then we can move alongside a direction going though point 0, 1, ... N-1. The next step in
/// the same direction goes back to the point at 0.
///
/// This structure is used for operation on [`LatticePoint`], [`LatticeLink`] and
/// [`LatticeLinkCanonical`].
// For example, theses three structures are abstract and are in general use to
/// access data on the lattice. These data are stored [`LinkMatrix`] and [`EField`] which are just
/// a wrapper around a [`Vec`]. `LatticeCyclic` is used to convert the lattice element to
/// an index to access these data.
///
/// This contain very few data and can be cloned at almost no cost even though
/// it does not implement [`Copy`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeCyclic<const D: usize> {
    /// The lattice spacing.
    size: Real,
    /// The number of point *per* dimension.
    dim: usize,
}

impl<const D: usize> LatticeCyclic<D> {
    /// Number space + time dimension, this is the `D` parameter.
    ///
    /// Not to confuse with [`LatticeCyclic::dim`] which is the number of point per dimension.
    pub const fn dim_st() -> usize {
        D
    }

    /// see [`LatticeLinkCanonical`], a conical link is a link whose direction is always positive.
    /// That means that a link form `[x, y, z, t]` with direction `-x`
    /// the link return is `[x - 1, y, z, t]` (modulo the `lattice::dim()``) with direction `+x`
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclic, DirectionEnum, LatticePoint, LatticeLinkCanonical};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let point = LatticePoint::<4>::new([1, 0, 2, 0].into());
    /// assert_eq!(
    ///     lattice.link_canonical(point, DirectionEnum::XNeg.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([0, 0, 2, 0].into()), DirectionEnum::XPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     lattice.link_canonical(point, DirectionEnum::XPos.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([1, 0, 2, 0].into()), DirectionEnum::XPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     lattice.link_canonical(point, DirectionEnum::YNeg.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([1, 3, 2, 0].into()), DirectionEnum::YPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn link_canonical(
        &self,
        pos: LatticePoint<D>,
        dir: Direction<D>,
    ) -> LatticeLinkCanonical<D> {
        let mut pos_link = pos;
        if !dir.is_positive() {
            pos_link = self.add_point_direction(pos_link, &dir);
        }
        for i in 0..pos.len() {
            pos_link[i] %= self.dim();
        }
        LatticeLinkCanonical::new(pos_link, dir.to_positive()).unwrap()
    }

    /// Return a link build from `pos` and `dir`.
    ///
    /// It is similar to [`LatticeLink::new`]. It however enforce that the point is inside the bounds.
    /// If it is not, it will use the modulus of the bound.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{lattice::{LatticeCyclic, Direction, LatticePoint}, error::ImplementationError};
    /// # use nalgebra::SVector;
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let l = LatticeCyclic::<3>::new(1_f64, 4)?;
    /// let dir = Direction::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let pt = LatticePoint::new(SVector::<_, 3>::new(1, 2, 5));
    /// let link = l.link(pt, dir);
    /// assert_eq!(
    ///     *link.pos(),
    ///     LatticePoint::new(SVector::<_, 3>::new(1, 2, 1))
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn link(&self, pos: LatticePoint<D>, dir: Direction<D>) -> LatticeLink<D> {
        let mut pos_link = LatticePoint::new_zero();
        for i in 0..pos.len() {
            pos_link[i] = pos[i] % self.dim();
        }
        LatticeLink::new(pos_link, dir)
    }

    /// Transform a [`LatticeLink`] into a [`LatticeLinkCanonical`].
    ///
    /// See [`LatticeCyclic::link_canonical`] and [`LatticeLinkCanonical`].
    pub fn into_canonical(&self, l: LatticeLink<D>) -> LatticeLinkCanonical<D> {
        self.link_canonical(*l.pos(), *l.dir())
    }

    /// Get the number of points in a single direction.
    ///
    /// use [`LatticeCyclic::number_of_points`] for the total number of points.
    /// Not to confuse with [`LatticeCyclic::dim_st`] which is the dimension of space-time.
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Get an Iterator over all points of the lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// for i in [4, 8, 16, 32, 64].into_iter() {
    ///     let l = LatticeCyclic::<4>::new(1_f64, i)?;
    ///     assert_eq!(l.get_points().size_hint().0, l.number_of_points());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_points(&self) -> IteratorLatticePoint<'_, D> {
        IteratorLatticePoint::new(self, LatticePoint::new_zero())
    }

    /// Get an Iterator over all canonical link of the lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// for i in [4, 8, 16, 32, 64].into_iter() {
    ///     let l = LatticeCyclic::<4>::new(1_f64, i)?;
    ///     assert_eq!(
    ///         l.get_links().size_hint().0,
    ///         l.number_of_canonical_links_space()
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_links(&self) -> IteratorLatticeLinkCanonical<'_, D> {
        return IteratorLatticeLinkCanonical::new(
            self,
            self.link_canonical(
                LatticePoint::new_zero(),
                *Direction::positive_directions().first().unwrap(),
            ),
        );
    }

    /// create a new lattice with `size` the lattice size parameter, and `dim` the number of
    /// points in each spatial dimension.
    ///
    /// # Errors
    /// Size should be greater than 0 and dim greater or equal to 2, otherwise return an error.
    pub fn new(size: Real, dim: usize) -> Result<Self, LatticeInitializationError> {
        if D == 0 {
            return Err(LatticeInitializationError::ZeroDimension);
        }
        if size <= 0_f64 || size.is_nan() || size.is_infinite() {
            return Err(LatticeInitializationError::NonPositiveSize);
        }
        if dim < 2 {
            return Err(LatticeInitializationError::DimTooSmall);
        }
        Ok(Self { size, dim })
    }

    /// Total number of canonical links oriented in space for a set time.
    ///
    /// Basically the number of element return by [`LatticeCyclic::get_links`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let l = LatticeCyclic::<4>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_canonical_links_space(), 8_usize.pow(4) * 4);
    ///
    /// let l = LatticeCyclic::<4>::new(1_f64, 16)?;
    /// assert_eq!(l.number_of_canonical_links_space(), 16_usize.pow(4) * 4);
    ///
    /// let l = LatticeCyclic::<3>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_canonical_links_space(), 8_usize.pow(3) * 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn number_of_canonical_links_space(&self) -> usize {
        self.number_of_points() * D
    }

    /// Total number of point in the lattice for a set time.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let l = LatticeCyclic::<4>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_points(), 8_usize.pow(4));
    ///
    /// let l = LatticeCyclic::<4>::new(1_f64, 16)?;
    /// assert_eq!(l.number_of_points(), 16_usize.pow(4));
    ///
    /// let l = LatticeCyclic::<3>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_points(), 8_usize.pow(3));
    /// # Ok(())
    /// # }
    /// ```
    pub fn number_of_points(&self) -> usize {
        self.dim().pow(D.try_into().unwrap())
    }

    /// Return the lattice size factor.
    pub const fn size(&self) -> Real {
        self.size
    }

    /// Get the next point in the lattice following the direction `dir`.
    /// It follows the Cyclic property of the lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclic, DirectionEnum, LatticePoint};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let point = LatticePoint::<4>::from([1, 0, 2, 0]);
    /// assert_eq!(
    ///     lattice.add_point_direction(point, &DirectionEnum::XPos.into()),
    ///     LatticePoint::from([2, 0, 2, 0])
    /// );
    /// // In the following case we get [_, 3, _, _] because `dim = 4`, and this lattice is Cyclic.
    /// assert_eq!(
    ///     lattice.add_point_direction(point, &DirectionEnum::YNeg.into()),
    ///     LatticePoint::from([1, 3, 2, 0])
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_point_direction(
        &self,
        point: LatticePoint<D>,
        dir: &Direction<D>,
    ) -> LatticePoint<D> {
        self.add_point_direction_n(point, dir, 1)
    }

    /// Returns the point given y moving `shift_number` times in direction `dir` from position `point`.
    /// It follows the Cyclic property of the lattice.
    ///
    /// It is equivalent of doing [`Self::add_point_direction`] n times.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclic, DirectionEnum, LatticePoint};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let point = LatticePoint::<4>::from([1, 0, 2, 0]);
    /// assert_eq!(
    ///     lattice.add_point_direction_n(point, &DirectionEnum::XPos.into(), 2),
    ///     LatticePoint::from([3, 0, 2, 0])
    /// );
    /// // In the following case we get [_, 1, _, _] because `dim = 4`, and this lattice is Cyclic.
    /// assert_eq!(
    ///     lattice.add_point_direction_n(point, &DirectionEnum::YNeg.into(), 3),
    ///     LatticePoint::from([1, 1, 2, 0])
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_point_direction_n(
        &self,
        mut point: LatticePoint<D>,
        dir: &Direction<D>,
        shift_number: usize,
    ) -> LatticePoint<D> {
        let shift_number = shift_number % self.dim(); // we ensure that shift_number < % self.dim()
        if dir.is_positive() {
            point[dir.index()] = (point[dir.index()] + shift_number) % self.dim();
        }
        else {
            let dir_pos = dir.to_positive();
            if point[dir_pos.index()] < shift_number {
                point[dir_pos.index()] = self.dim() - (shift_number - point[dir_pos.index()]);
            }
            else {
                point[dir_pos.index()] = (point[dir_pos.index()] - shift_number) % self.dim();
            }
        }
        point
    }

    /// Returns whether the number of canonical link is the same as the length of `links`.
    pub fn has_compatible_length_links(&self, links: &LinkMatrix) -> bool {
        self.number_of_canonical_links_space() == links.len()
    }

    /// Returns wether the number of point is the same as the length of `e_field`.
    pub fn has_compatible_length_e_field(&self, e_field: &EField<D>) -> bool {
        self.number_of_points() == e_field.len()
    }

    /// Returns the length is compatible for both `links` and `e_field`.
    /// See [`Self::has_compatible_length_links`] and [`Self::has_compatible_length_e_field`].
    pub fn has_compatible_length(&self, links: &LinkMatrix, e_field: &EField<D>) -> bool {
        self.has_compatible_length_links(links) && self.has_compatible_length_e_field(e_field)
    }
}

impl<const D: usize> std::fmt::Display for LatticeCyclic<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cyclic lattice with {}^{} points and spacing {}",
            self.dim, D, self.size
        )
    }
}

/// Iterator over [`LatticeLinkCanonical`] associated to a particular [`LatticeCyclic`].
#[derive(Clone, Debug, PartialEq)]
pub struct IteratorLatticeLinkCanonical<'a, const D: usize> {
    lattice: &'a LatticeCyclic<D>,
    element: Option<LatticeLinkCanonical<D>>,
}

impl<'a, const D: usize> IteratorLatticeLinkCanonical<'a, D> {
    /// create a new iterator. The first [`IteratorLatticeLinkCanonical::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticeLinkCanonical, LatticeCyclic, LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let first_el = LatticeLinkCanonical::<4>::new(LatticePoint::from([1, 0, 2, 0]), DirectionEnum::YPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let mut iter = IteratorLatticeLinkCanonical::new(&lattice, first_el);
    /// assert_eq!(iter.next().ok_or(ImplementationError::OptionWithUnexpectedNone)?, first_el);
    /// # Ok(())
    /// # }
    /// ```
    pub const fn new(lattice: &'a LatticeCyclic<D>, first_el: LatticeLinkCanonical<D>) -> Self {
        Self {
            lattice,
            element: Some(first_el),
        }
    }
}

impl<'a, const D: usize> Iterator for IteratorLatticeLinkCanonical<'a, D> {
    type Item = LatticeLinkCanonical<D>;

    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        let previous_el = self.element;
        if let Some(ref mut element) = self.element {
            let mut iter_dir = IteratorDirection::<D, true>::new(Some(element.dir)).unwrap();
            let new_dir = iter_dir.next();
            match new_dir {
                Some(dir) => element.set_dir(dir),
                None => {
                    element.set_dir(Direction::new(0, true).unwrap());
                    let mut iter = IteratorLatticePoint::new(self.lattice, *element.pos());
                    match iter.nth(1) {
                        // get the second element
                        Some(array) => *element.pos_mut() = array,
                        None => {
                            self.element = None;
                            return previous_el;
                        }
                    }
                }
            }
        }
        previous_el
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.element {
            None => (0, Some(0)),
            Some(element) => {
                let val =
                    self.lattice.number_of_canonical_links_space() - element.to_index(self.lattice);
                (val, Some(val))
            }
        }
    }
}

impl<'a, const D: usize> FusedIterator for IteratorLatticeLinkCanonical<'a, D> {}

impl<'a, const D: usize> ExactSizeIterator for IteratorLatticeLinkCanonical<'a, D> {}

/// Enum for internal use of iterator. It store the previous element returned by `next`
#[derive(Clone, Debug, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum IteratorElement<T> {
    /// First element of the iterator
    FirstElement,
    /// An element of the iterator
    Element(T),
    /// The Iterator is exhausted
    LastElement,
}

impl<T: std::fmt::Display> std::fmt::Display for IteratorElement<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstElement => write!(f, "first element"),
            Self::Element(t) => write!(f, "element {}", t),
            Self::LastElement => write!(f, "last element"),
        }
    }
}

impl<T> Default for IteratorElement<T> {
    fn default() -> Self {
        Self::FirstElement
    }
}

impl<T> From<IteratorElement<T>> for Option<T> {
    fn from(element: IteratorElement<T>) -> Self {
        match element {
            IteratorElement::Element(el) => Some(el),
            IteratorElement::LastElement | IteratorElement::FirstElement => None,
        }
    }
}

/// Iterator over [`Direction`] with the same sign.
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::{IteratorDirection, Direction, IteratorElement};
/// # use lattice_qcd_rs::error::ImplementationError;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut iter = IteratorDirection::<4, true>::new(None)
///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     Direction::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     Direction::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     Direction::new(2, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     Direction::new(3, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(iter.next(), None);
/// assert_eq!(iter.next(), None);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct IteratorDirection<const D: usize, const IS_POSITIVE_DIRECTION: bool> {
    element: IteratorElement<Direction<D>>,
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool>
    IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    /// Create an iterator where the first element upon calling [`Self::next`] is the direction
    /// after `element`. Giving `None` results in the first element being the direction with index 0
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorDirection, Direction, IteratorElement};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut iter = IteratorDirection::<4, true>::new(None)
    ///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// let element =
    ///     Direction::<4>::new(0, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let mut iter = IteratorDirection::<4, false>::new(Some(element))
    ///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(2, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorDirection, Direction, IteratorElement};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let iter = IteratorDirection::<0, true>::new(None);
    /// // 0 is invalid
    /// assert!(iter.is_none());
    /// let iter = IteratorDirection::<4, true>::new(Some(
    ///     Direction::new(2, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    /// ));
    /// // the sign of the direction is invalid
    /// assert!(iter.is_none());
    /// # Ok(())
    /// # }
    /// ```
    pub const fn new(element: Option<Direction<D>>) -> Option<Self> {
        match element {
            None => Self::new_from_element(IteratorElement::FirstElement),
            Some(dir) => Self::new_from_element(IteratorElement::Element(dir)),
        }
    }

    /// create a new iterator. The first [`IteratorLatticeLinkCanonical::next()`] the element just after the one given
    /// or the first element if [`IteratorElement::FirstElement`] is given.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorDirection, Direction, IteratorElement};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut iter = IteratorDirection::<4, true>::new_from_element(IteratorElement::FirstElement)
    ///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// let element =
    ///     Direction::<4>::new(0, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let mut iter =
    ///     IteratorDirection::<4, false>::new_from_element(IteratorElement::Element(element))
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     Direction::new(2, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub const fn new_from_element(element: IteratorElement<Direction<D>>) -> Option<Self> {
        if D == 0 {
            return None;
        }
        if let IteratorElement::Element(ref el) = element {
            if el.is_positive() != IS_POSITIVE_DIRECTION {
                return None;
            }
        }
        Some(Self { element })
    }
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> Iterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    type Item = Direction<D>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_element = match self.element {
            IteratorElement::FirstElement => {
                IteratorElement::Element(Direction::new(0, IS_POSITIVE_DIRECTION).unwrap())
            }
            IteratorElement::Element(ref dir) => {
                if let Some(dir) = Direction::new(dir.index() + 1, IS_POSITIVE_DIRECTION) {
                    IteratorElement::Element(dir)
                }
                else {
                    IteratorElement::LastElement
                }
            }
            IteratorElement::LastElement => IteratorElement::LastElement,
        };
        self.element = next_element;
        next_element.into()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = match self.element {
            IteratorElement::FirstElement => D,
            IteratorElement::Element(ref dir) => D - (dir.index() + 1),
            IteratorElement::LastElement => 0,
        };
        (size, Some(size))
    }
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> FusedIterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> ExactSizeIterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
}

/// Iterator over [`LatticePoint`]
#[derive(Clone, Debug, PartialEq)]
pub struct IteratorLatticePoint<'a, const D: usize> {
    lattice: &'a LatticeCyclic<D>,
    element: Option<LatticePoint<D>>,
}

impl<'a, const D: usize> IteratorLatticePoint<'a, D> {
    /// create a new iterator. The first [`IteratorLatticePoint::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticePoint, LatticeCyclic, LatticePoint, Direction};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::new(1_f64, 4)?;
    /// let first_el = LatticePoint::from([1, 0, 2, 0]);
    /// let mut iter = IteratorLatticePoint::new(&lattice, first_el);
    /// assert_eq!(iter.next().ok_or(ImplementationError::OptionWithUnexpectedNone)?, first_el);
    /// # Ok(())
    /// # }
    /// ```
    pub const fn new(lattice: &'a LatticeCyclic<D>, first_el: LatticePoint<D>) -> Self {
        Self {
            lattice,
            element: Some(first_el),
        }
    }
}

impl<'a, const D: usize> Iterator for IteratorLatticePoint<'a, D> {
    type Item = LatticePoint<D>;

    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        let previous_el = self.element;
        if let Some(ref mut el) = &mut self.element {
            el[0] += 1;
            for i in 0..el.len() {
                while el[i] >= self.lattice.dim() {
                    if i < el.len() - 1 {
                        // every element except the last one
                        el[i + 1] += 1;
                    }
                    else {
                        self.element = None;
                        return previous_el;
                    }
                    el[i] -= self.lattice.dim();
                }
            }
        }
        previous_el
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.element {
            None => (0, Some(0)),
            Some(element) => {
                let val = self.lattice.number_of_points() - element.to_index(self.lattice);
                (val, Some(val))
            }
        }
    }
}

impl<'a, const D: usize> FusedIterator for IteratorLatticePoint<'a, D> {}

impl<'a, const D: usize> ExactSizeIterator for IteratorLatticePoint<'a, D> {}

/// Represents point on a (any) lattice.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticePoint<const D: usize> {
    data: na::SVector<usize, D>,
}

impl<const D: usize> LatticePoint<D> {
    /// Create a new lattice point.
    ///
    /// It can be outside a lattice.
    pub const fn new(data: SVector<usize, D>) -> Self {
        Self { data }
    }

    /// Create a point at the origin
    pub fn new_zero() -> Self {
        Self {
            data: SVector::zeros(),
        }
    }

    /// Create a point using the closure generate elements with the index as input.
    ///
    /// See [`nalgebra::base::Matrix::from_fn`].
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> usize,
    {
        Self::new(SVector::from_fn(|index, _| f(index)))
    }

    /// Number of elements in [`LatticePoint`]. This is `D`.
    #[allow(clippy::unused_self)]
    pub const fn len(&self) -> usize {
        // this is in order to have a const function.
        // we could have called self.data.len()
        D
    }

    /// Return if LatticePoint contain no data. True when the dimension is 0, false otherwise.
    #[allow(clippy::unused_self)]
    pub const fn is_empty(&self) -> bool {
        D == 0
    }

    /// Get an iterator on the data.
    pub fn iter(&self) -> impl Iterator<Item = &usize> + ExactSizeIterator + FusedIterator {
        self.data.iter()
    }

    /// Get an iterator on the data as mutable.
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
    pub fn as_svector_mut(&mut self) -> &mut SVector<usize, D> {
        &mut self.data
    }
}

impl<const D: usize> Default for LatticePoint<D> {
    fn default() -> Self {
        Self::new_zero()
    }
}

impl<const D: usize> std::fmt::Display for LatticePoint<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<'a, const D: usize> IntoIterator for &'a LatticePoint<D> {
    type IntoIter = <&'a SVector<usize, D> as IntoIterator>::IntoIter;
    type Item = &'a usize;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a mut LatticePoint<D> {
    type IntoIter = <&'a mut SVector<usize, D> as IntoIterator>::IntoIter;
    type Item = &'a mut usize;

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
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output {
        &mut self.data[pos]
    }
}

impl<T, const D: usize> From<T> for LatticePoint<D>
where
    SVector<usize, D>: From<T>,
{
    fn from(data: T) -> Self {
        LatticePoint::new(SVector::from(data))
    }
}

impl<const D: usize> From<LatticePoint<D>> for [usize; D]
where
    SVector<usize, D>: Into<[usize; D]>,
{
    fn from(data: LatticePoint<D>) -> [usize; D] {
        data.data.into()
    }
}

impl<const D: usize> AsRef<SVector<usize, D>> for LatticePoint<D> {
    fn as_ref(&self) -> &SVector<usize, D> {
        self.as_svector()
    }
}

impl<const D: usize> AsMut<SVector<usize, D>> for LatticePoint<D> {
    fn as_mut(&mut self) -> &mut SVector<usize, D> {
        self.as_svector_mut()
    }
}

/// Trait to convert an element on a lattice to an [`usize`].
///
/// Used mainly to index field on the lattice using [`std::vec::Vec`]
// TODO change name ? make it less confusing
pub trait LatticeElementToIndex<const D: usize> {
    /// Given a lattice return an index from the element
    fn to_index(&self, l: &LatticeCyclic<D>) -> usize;
}

impl<const D: usize> LatticeElementToIndex<D> for LatticePoint<D> {
    fn to_index(&self, l: &LatticeCyclic<D>) -> usize {
        self.iter()
            .enumerate()
            .map(|(index, pos)| (pos % l.dim()) * l.dim().pow(index.try_into().unwrap()))
            .sum()
    }
}

impl<const D: usize> LatticeElementToIndex<D> for Direction<D> {
    /// equivalent to [`Direction::to_index()`]
    fn to_index(&self, _: &LatticeCyclic<D>) -> usize {
        self.index()
    }
}

impl<const D: usize> LatticeElementToIndex<D> for LatticeLinkCanonical<D> {
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
    fn to_index(&self, _l: &LatticeCyclic<D>) -> usize {
        *self
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeLinkCanonical<const D: usize> {
    from: LatticePoint<D>,
    dir: Direction<D>,
}

impl<const D: usize> LatticeLinkCanonical<D> {
    /// Try create a LatticeLinkCanonical. If the dir is negative it fails.
    ///
    /// To guaranty creating an element see [LatticeCyclic::link_canonical].
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
    pub const fn new(from: LatticePoint<D>, dir: Direction<D>) -> Option<Self> {
        if dir.is_negative() {
            return None;
        }
        Some(Self { from, dir })
    }

    /// Position of the link.
    pub const fn pos(&self) -> &LatticePoint<D> {
        &self.from
    }

    /// Get a mutable reference on the position of the link.
    pub fn pos_mut(&mut self) -> &mut LatticePoint<D> {
        &mut self.from
    }

    /// Direction of the link.
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
    /// # Panic
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
    pub fn set_dir(&mut self, dir: Direction<D>) {
        if dir.is_negative() {
            panic!("Cannot set a negative direction to a canonical link.");
        }
        self.dir = dir;
    }

    /// Set the direction using positive direction. i.e. if a direction `-x` is passed
    /// the direction assigned will be `+x`.
    ///
    /// This is equivalent to `link.set_dir(dir.to_positive())`.
    pub fn set_dir_positive(&mut self, dir: Direction<D>) {
        self.dir = dir.to_positive();
    }
}

impl<const D: usize> std::fmt::Display for LatticeLinkCanonical<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "canonical link [position {}, direction {}]",
            self.from, self.dir
        )
    }
}

impl<const D: usize> From<LatticeLinkCanonical<D>> for LatticeLink<D> {
    fn from(l: LatticeLinkCanonical<D>) -> Self {
        LatticeLink::new(l.from, l.dir)
    }
}

impl<const D: usize> From<&LatticeLinkCanonical<D>> for LatticeLink<D> {
    fn from(l: &LatticeLinkCanonical<D>) -> Self {
        LatticeLink::new(l.from, l.dir)
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
    pub const fn new(from: LatticePoint<D>, dir: Direction<D>) -> Self {
        Self { from, dir }
    }

    /// Get the position of the link.
    pub const fn pos(&self) -> &LatticePoint<D> {
        &self.from
    }

    /// Get a mutable reference to the position of the link.
    pub fn pos_mut(&mut self) -> &mut LatticePoint<D> {
        &mut self.from
    }

    /// Get the direction of the link.
    pub const fn dir(&self) -> &Direction<D> {
        &self.dir
    }

    /// Get a mutable reference to the direction of the link.
    pub fn dir_mut(&mut self) -> &mut Direction<D> {
        &mut self.dir
    }

    /// Get if the direction of the link is positive.
    pub const fn is_dir_positive(&self) -> bool {
        self.dir.is_positive()
    }

    /// Get if the direction of the link is negative.
    pub const fn is_dir_negative(&self) -> bool {
        self.dir.is_negative()
    }
}

impl<const D: usize> std::fmt::Display for LatticeLink<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "link [position {}, direction {}]", self.from, self.dir)
    }
}

/// Represent a cardinal direction
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Direction<const D: usize> {
    index_dir: usize,
    is_positive: bool,
}

impl<const D: usize> Direction<D> {
    /// New direction from a direction as an idex and wether it is in the positive direction.
    pub const fn new(index_dir: usize, is_positive: bool) -> Option<Self> {
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
    pub fn positives_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(D);
        for i in 0..D {
            x.push(Self::new(i, true).unwrap());
        }
        x
    }

    /// List all directions.
    /// This is very slow use [`DirectionList::directions`] instead.
    pub fn directions_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(2 * D);
        for i in 0..D {
            x.push(Self::new(i, true).unwrap());
            x.push(Self::new(i, false).unwrap());
        }
        x
    }

    // TODO add const function for all direction once operation on const generic are added
    /// Get all direction with the sign `IS_POSITIVE`.
    pub const fn directions_array<const IS_POSITIVE: bool>() -> [Self; D] {
        // TODO use unsafe code to avoid the allocation
        let mut i = 0_usize;
        let mut array = [Direction {
            index_dir: 0,
            is_positive: IS_POSITIVE,
        }; D];
        while i < D {
            array[i] = Direction {
                index_dir: i,
                is_positive: IS_POSITIVE,
            };
            i += 1;
        }
        array
    }

    /// Get all negative direction
    pub const fn negative_directions() -> [Self; D] {
        Self::directions_array::<false>()
    }

    /// Get all positive direction
    pub const fn positive_directions() -> [Self; D] {
        Self::directions_array::<true>()
    }

    /// Get if the position is positive.
    pub const fn is_positive(&self) -> bool {
        self.is_positive
    }

    /// Get if the position is Negative. see [`Direction::is_positive`]
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
    pub const fn to_positive(mut self) -> Self {
        self.is_positive = true;
        self
    }

    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// assert_eq!(Direction::<4>::new(1, false).unwrap().index(), 1);
    /// ```
    pub const fn index(&self) -> usize {
        self.index_dir
    }

    /// Convert the direction into a vector of norm `a`;
    pub fn to_vector(self, a: f64) -> SVector<Real, D> {
        self.to_unit_vector() * a
    }

    /// Returns the dimension
    pub const fn dim() -> usize {
        D
    }

    /// Convert the direction into a vector of norm `1`;
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

impl<const D: usize> std::fmt::Display for Direction<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[index {}, is positive {}]",
            self.index(),
            self.is_positive()
        )
    }
}

/// List all possible direction
pub trait DirectionList: Sized {
    /// List all directions.
    fn directions() -> &'static [Self];
    /// List all positive directions.
    fn positive_directions() -> &'static [Self];
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        }
        else if self.is_positive() == other.is_positive() {
            self.index().partial_cmp(&other.index())
        }
        else if self.index() == other.index() {
            self.is_positive().partial_cmp(&other.is_positive())
        }
        else {
            None
        }
    }
}

impl<const D: usize> Neg for Direction<D> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.is_positive = !self.is_positive;
        self
    }
}

impl<const D: usize> Neg for &Direction<D> {
    type Output = Direction<D>;

    fn neg(self) -> Self::Output {
        -*self
    }
}

/// Return [`Direction::to_index`].
impl<const D: usize> From<Direction<D>> for usize {
    fn from(d: Direction<D>) -> Self {
        d.index()
    }
}

/// Return [`Direction::to_index`].
impl<const D: usize> From<&Direction<D>> for usize {
    fn from(d: &Direction<D>) -> Self {
        d.index()
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<const D: usize> From<SVector<Real, D>> for Direction<D> {
    fn from(v: SVector<Real, D>) -> Self {
        Direction::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<const D: usize> From<&SVector<Real, D>> for Direction<D> {
    fn from(v: &SVector<Real, D>) -> Self {
        Direction::<D>::from_vector(v)
    }
}

/// Return [`Direction::to_unit_vector`].
impl<const D: usize> From<Direction<D>> for SVector<Real, D> {
    fn from(d: Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`Direction::to_unit_vector`].
impl<const D: usize> From<&Direction<D>> for SVector<Real, D> {
    fn from(d: &Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

impl From<DirectionEnum> for Direction<4> {
    fn from(d: DirectionEnum) -> Self {
        Self::new(d.to_index(), d.is_positive()).expect("unreachable")
    }
}

impl From<&DirectionEnum> for Direction<4> {
    fn from(d: &DirectionEnum) -> Self {
        Self::new(d.to_index(), d.is_positive()).expect("unreachable")
    }
}

/// Represent a cardinal direction
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
        DirectionEnum::XPos,
        DirectionEnum::YPos,
        DirectionEnum::ZPos,
        DirectionEnum::TPos,
        DirectionEnum::XNeg,
        DirectionEnum::YNeg,
        DirectionEnum::ZNeg,
        DirectionEnum::TNeg,
    ];
    /// List all spatial directions.
    pub const DIRECTIONS_SPACE: [Self; 6] = [
        DirectionEnum::XPos,
        DirectionEnum::YPos,
        DirectionEnum::ZPos,
        DirectionEnum::XNeg,
        DirectionEnum::YNeg,
        DirectionEnum::ZNeg,
    ];
    /// List of all positives directions.
    pub const POSITIVES: [Self; 4] = [
        DirectionEnum::XPos,
        DirectionEnum::YPos,
        DirectionEnum::ZPos,
        DirectionEnum::TPos,
    ];
    /// List spatial positive direction.
    pub const POSITIVES_SPACE: [Self; 3] = [
        DirectionEnum::XPos,
        DirectionEnum::YPos,
        DirectionEnum::ZPos,
    ];

    /// Convert the direction into a vector of norm `a`;
    pub fn to_vector(self, a: f64) -> Vector4<Real> {
        self.to_unit_vector() * a
    }

    /// Convert the direction into a vector of norm `1`;
    pub const fn to_unit_vector(self) -> Vector4<Real> {
        match self {
            DirectionEnum::XPos => Vector4::<Real>::new(1_f64, 0_f64, 0_f64, 0_f64),
            DirectionEnum::XNeg => Vector4::<Real>::new(-1_f64, 0_f64, 0_f64, 0_f64),
            DirectionEnum::YPos => Vector4::<Real>::new(0_f64, 1_f64, 0_f64, 0_f64),
            DirectionEnum::YNeg => Vector4::<Real>::new(0_f64, -1_f64, 0_f64, 0_f64),
            DirectionEnum::ZPos => Vector4::<Real>::new(0_f64, 0_f64, 1_f64, 0_f64),
            DirectionEnum::ZNeg => Vector4::<Real>::new(0_f64, 0_f64, -1_f64, 0_f64),
            DirectionEnum::TPos => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, 1_f64),
            DirectionEnum::TNeg => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, -1_f64),
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
    pub const fn is_positive(self) -> bool {
        match self {
            DirectionEnum::XPos
            | DirectionEnum::YPos
            | DirectionEnum::ZPos
            | DirectionEnum::TPos => true,
            DirectionEnum::XNeg
            | DirectionEnum::YNeg
            | DirectionEnum::ZNeg
            | DirectionEnum::TNeg => false,
        }
    }

    /// Get if the position is Negative. see [`DirectionEnum::is_positive`]
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
    #[allow(clippy::needless_return)] // for readability
    pub fn from_vector(v: &Vector4<Real>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for i in 0..DirectionEnum::POSITIVES.len() {
            let scalar_prod = v.dot(&DirectionEnum::POSITIVES[i].to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        match index_max {
            0 => {
                if is_positive {
                    return DirectionEnum::XPos;
                }
                else {
                    return DirectionEnum::XNeg;
                }
            }
            1 => {
                if is_positive {
                    return DirectionEnum::YPos;
                }
                else {
                    return DirectionEnum::YNeg;
                }
            }
            2 => {
                if is_positive {
                    return DirectionEnum::ZPos;
                }
                else {
                    return DirectionEnum::ZNeg;
                }
            }
            3 => {
                if is_positive {
                    return DirectionEnum::TPos;
                }
                else {
                    return DirectionEnum::TNeg;
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
    pub const fn to_positive(self) -> Self {
        match self {
            DirectionEnum::XNeg => DirectionEnum::XPos,
            DirectionEnum::YNeg => DirectionEnum::YPos,
            DirectionEnum::ZNeg => DirectionEnum::ZPos,
            DirectionEnum::TNeg => DirectionEnum::TPos,
            _ => self,
        }
    }

    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XPos.to_index(), 0);
    /// assert_eq!(DirectionEnum::XNeg.to_index(), 0);
    /// assert_eq!(DirectionEnum::YPos.to_index(), 1);
    /// assert_eq!(DirectionEnum::YNeg.to_index(), 1);
    /// assert_eq!(DirectionEnum::ZPos.to_index(), 2);
    /// assert_eq!(DirectionEnum::ZNeg.to_index(), 2);
    /// assert_eq!(DirectionEnum::TPos.to_index(), 3);
    /// assert_eq!(DirectionEnum::TNeg.to_index(), 3);
    /// ```
    pub const fn to_index(self) -> usize {
        match self {
            DirectionEnum::XPos | DirectionEnum::XNeg => 0,
            DirectionEnum::YPos | DirectionEnum::YNeg => 1,
            DirectionEnum::ZPos | DirectionEnum::ZNeg => 2,
            DirectionEnum::TPos | DirectionEnum::TNeg => 3,
        }
    }
}

/// Return [`DirectionEnum::XPos`]
impl Default for DirectionEnum {
    ///Return [`DirectionEnum::XPos`]
    fn default() -> Self {
        Self::XPos
    }
}

impl std::fmt::Display for DirectionEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirectionEnum::XPos => write!(f, "positive X direction"),
            DirectionEnum::XNeg => write!(f, "negative X direction"),
            DirectionEnum::YPos => write!(f, "positive Y direction"),
            DirectionEnum::YNeg => write!(f, "negative Y direction"),
            DirectionEnum::ZPos => write!(f, "positive Z direction"),
            DirectionEnum::ZNeg => write!(f, "negative Z direction"),
            DirectionEnum::TPos => write!(f, "positive T direction"),
            DirectionEnum::TNeg => write!(f, "negative T direction"),
        }
    }
}

impl DirectionList for DirectionEnum {
    fn directions() -> &'static [Self] {
        &Self::DIRECTIONS
    }

    fn positive_directions() -> &'static [Self] {
        &Self::POSITIVES
    }
}

impl PartialOrd for DirectionEnum {
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

    fn neg(self) -> Self::Output {
        match self {
            DirectionEnum::XPos => DirectionEnum::XNeg,
            DirectionEnum::XNeg => DirectionEnum::XPos,
            DirectionEnum::YPos => DirectionEnum::YNeg,
            DirectionEnum::YNeg => DirectionEnum::YPos,
            DirectionEnum::ZPos => DirectionEnum::ZNeg,
            DirectionEnum::ZNeg => DirectionEnum::ZPos,
            DirectionEnum::TPos => DirectionEnum::TNeg,
            DirectionEnum::TNeg => DirectionEnum::TPos,
        }
    }
}

impl Neg for &DirectionEnum {
    type Output = Self;

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
    fn from(d: DirectionEnum) -> Self {
        d.to_index()
    }
}

/// Return [`DirectionEnum::to_index`].
impl From<&DirectionEnum> for usize {
    fn from(d: &DirectionEnum) -> Self {
        d.to_index()
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<Vector4<Real>> for DirectionEnum {
    fn from(v: Vector4<Real>) -> Self {
        DirectionEnum::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<&Vector4<Real>> for DirectionEnum {
    fn from(v: &Vector4<Real>) -> Self {
        DirectionEnum::from_vector(v)
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<DirectionEnum> for Vector4<Real> {
    fn from(d: DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<&DirectionEnum> for Vector4<Real> {
    fn from(d: &DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn directions_list_cst() {
        let dirs = Direction::<3>::positive_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(*dir, Direction::new(index, true).unwrap());
        }
        let dirs = Direction::<3>::negative_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(*dir, Direction::new(index, false).unwrap());
        }

        let dirs = Direction::<8>::positive_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(*dir, Direction::new(index, true).unwrap());
        }
        let dirs = Direction::<8>::negative_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(*dir, Direction::new(index, false).unwrap());
        }

        let dirs = Direction::<0>::positive_directions();
        assert_eq!(dirs.len(), 0);
        let dirs = Direction::<0>::negative_directions();
        assert_eq!(dirs.len(), 0);

        let dirs = Direction::<128>::positive_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(*dir, Direction::new(index, true).unwrap());
        }
        let dirs = Direction::<128>::negative_directions();
        for (index, dir) in dirs.iter().enumerate() {
            assert_eq!(*dir, Direction::new(index, false).unwrap());
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
    #[should_panic]
    fn set_dir_neg() {
        let mut lattice_link_canonical =
            LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XPos.into())
                .unwrap();

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

        let l = LatticeCyclic::new(1_f64, 4).unwrap();
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

    #[test]
    fn lattice_link() {
        let pt = [0, 0, 0, 0].into();
        let dir = DirectionEnum::XNeg.into();
        let mut link = LatticeLinkCanonical::new(pt, DirectionEnum::YPos.into()).unwrap();
        link.set_dir_positive(dir);
        assert_eq!(
            link,
            LatticeLinkCanonical::new(pt, dir.to_positive()).unwrap()
        );

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
            Direction::new(0, true).unwrap(),
        )
        .unwrap();
        println!("{}", canonical_link);
        assert_eq!(
            canonical_link.to_string(),
            format!(
                "canonical link [position {}, direction [index 0, is positive true]]",
                vector
            )
        );
    }

    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn iterator() {
        let l = LatticeCyclic::<2>::new(1_f64, 4).unwrap();
        let mut iterator = l.get_points();
        assert_eq!(
            iterator.size_hint(),
            (l.number_of_points(), Some(l.number_of_points()))
        );
        assert_eq!(iterator.size_hint(), (16, Some(16)));
        iterator.nth(9);
        assert_eq!(
            iterator.size_hint(),
            (
                l.number_of_points() - 10,       // 6
                Some(l.number_of_points() - 10)  // 6
            )
        );
        assert!(iterator.nth(4).is_some());
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert!(iterator.next().is_none());
        assert_eq!(iterator.size_hint(), (0, Some(0)));

        let mut iterator = l.get_links();
        assert_eq!(
            iterator.size_hint(),
            (
                l.number_of_canonical_links_space(),
                Some(l.number_of_canonical_links_space())
            )
        );
        assert_eq!(iterator.size_hint(), (16 * 2, Some(16 * 2)));
        iterator.nth(9);
        assert_eq!(
            iterator.size_hint(),
            (
                l.number_of_canonical_links_space() - 10,       // 6
                Some(l.number_of_canonical_links_space() - 10)  // 6
            )
        );
        assert!(iterator.nth(20).is_some());
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert!(iterator.next().is_none());
        assert_eq!(iterator.size_hint(), (0, Some(0)));

        let mut iterator = IteratorDirection::<2, true>::new(None).unwrap();
        assert_eq!(iterator.size_hint(), (2, Some(2)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert!(iterator.next().is_none());
        assert!(iterator.next().is_none());

        //----

        assert_eq!(
            IteratorElement::<i32>::FirstElement.to_string(),
            "first element"
        );
        assert_eq!(IteratorElement::Element(0_i32).to_string(), "element 0");
        assert_eq!(
            IteratorElement::<i32>::LastElement.to_string(),
            "last element"
        );
        assert_eq!(
            IteratorElement::<i32>::default(),
            IteratorElement::<i32>::FirstElement,
        );
    }

    #[test]
    fn lattice() {
        let lattice = LatticeCyclic::<3>::new(1_f64, 8).unwrap();
        assert_eq!(
            lattice.to_string(),
            "Cyclic lattice with 8^3 points and spacing 1"
        );
    }
}
