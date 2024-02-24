//! Contains [`IteratorElement`]

//---------------------------------------
// uses

use std::fmt::{self, Display};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::Sealed;

use crate::lattice::{
    direction::DirectionIndexing, LatticeCyclic, LatticeElementToIndex, NumberOfLatticeElement,
};

//---------------------------------------
// enum definition

/// Enum for internal use of iterator. It store the previous element returned by `next`
#[allow(clippy::exhaustive_enums)]
#[derive(Sealed, Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum IteratorElement<T> {
    /// First element of the iterator
    FirstElement,
    /// An element of the iterator
    Element(T),
    /// The Iterator is exhausted
    LastElement,
}

//---------------------------------------
// impl block

impl<T> IteratorElement<T> {
    // TODO useful fn like map or other think like Option

    /// Map an [`IteratorElement::<T>`] to [`IteratorElement::<U>`]
    #[inline]
    #[must_use]
    pub fn map<U, F: FnOnce(T) -> U>(self, func: F) -> IteratorElement<U> {
        match self {
            Self::FirstElement => IteratorElement::FirstElement,
            Self::Element(element) => IteratorElement::Element(func(element)),
            Self::LastElement => IteratorElement::LastElement,
        }
    }

    /// Convert a `&IteratorElement<T>` to an `IteratorElement<&T>`.
    #[inline]
    #[must_use]
    pub const fn as_ref(&self) -> IteratorElement<&T> {
        match self {
            Self::FirstElement => IteratorElement::FirstElement,
            Self::Element(ref t) => IteratorElement::Element(t),
            Self::LastElement => IteratorElement::LastElement,
        }
    }

    /// Convert a `&mut IteratorElement<T>` to an `IteratorElement<&mut T>`.
    #[inline]
    #[must_use]
    pub fn as_mut(&mut self) -> IteratorElement<&mut T> {
        match self {
            Self::FirstElement => IteratorElement::FirstElement,
            Self::Element(ref mut t) => IteratorElement::Element(t),
            Self::LastElement => IteratorElement::LastElement,
        }
    }

    /// Transform an index to an element mapping o to [`Self::FirstElement`] and
    /// index - 1 to the closure. If the closure returns None [`Self::LastElement`] is
    /// returned instead.
    #[inline]
    #[must_use]
    pub(super) fn index_to_element<F: FnOnce(usize) -> Option<T>>(
        index: usize,
        closure: F,
    ) -> Self {
        if index == 0 {
            Self::FirstElement
        } else {
            closure(index - 1).map_or(Self::LastElement, Self::Element)
        }
    }

    /// Computes the number of elements left with the given element as the `front`
    /// of an potential iterator.
    #[must_use]
    #[inline]
    pub(super) fn size_position<const D: usize>(&self, lattice: &LatticeCyclic<D>) -> usize
    where
        T: LatticeElementToIndex<D> + NumberOfLatticeElement<D>,
    {
        match self {
            Self::FirstElement => T::number_of_elements(lattice),
            Self::Element(ref element) => {
                T::number_of_elements(lattice).saturating_sub(element.to_index(lattice) + 1)
            }
            Self::LastElement => 0,
        }
    }
}

//---------------------------------------
// common traits

impl<T: Display> Display for IteratorElement<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstElement => write!(f, "first element"),
            Self::Element(t) => write!(f, "element {t}"),
            Self::LastElement => write!(f, "last element"),
        }
    }
}

/// Returns [`IteratorElement::FirstElement`]
impl<T> Default for IteratorElement<T> {
    #[inline]
    fn default() -> Self {
        Self::FirstElement
    }
}

//---------------------------------------
// conversion traits

/// Returns [`Some`] in the case of [`IteratorElement::Element`], [`None`] otherwise.
impl<T> From<IteratorElement<T>> for Option<T> {
    #[inline]
    fn from(element: IteratorElement<T>) -> Self {
        match element {
            IteratorElement::Element(el) => Some(el),
            IteratorElement::LastElement | IteratorElement::FirstElement => None,
        }
    }
}

//---------------------------------------
// Lattice indexing

/// Index [`Self`] as 0 for the [`Self::FirstElement`],
/// [`NumberOfLatticeElement::number_of_elements`] + 1 for [`Self::LastElement`]
/// and index + 1 for any other element.
impl<const D: usize, T: LatticeElementToIndex<D> + NumberOfLatticeElement<D>>
    LatticeElementToIndex<D> for IteratorElement<T>
{
    #[inline]
    fn to_index(&self, lattice: &LatticeCyclic<D>) -> usize {
        match self {
            Self::FirstElement => 0,
            Self::Element(ref element) => element.to_index(lattice) + 1,
            Self::LastElement => T::number_of_elements(lattice) + 1,
        }
    }
}

/// The number of element is the number of element of `T` + 2
impl<const D: usize, T: LatticeElementToIndex<D> + NumberOfLatticeElement<D>>
    NumberOfLatticeElement<D> for IteratorElement<T>
{
    #[inline]
    fn number_of_elements(lattice: &LatticeCyclic<D>) -> usize {
        T::number_of_elements(lattice) + 2
    }
}

//---------------------------------------
// DirectionIndexing

impl<D: DirectionIndexing> DirectionIndexing for IteratorElement<D> {
    fn direction_to_index(&self) -> usize {
        match self {
            Self::FirstElement => 0,
            Self::Element(ref e) => e.direction_to_index() + 1,
            Self::LastElement => D::number_of_directions() + 1,
        }
    }

    fn direction_from_index(index: usize) -> Option<Self> {
        (index < Self::number_of_directions())
            .then(|| Self::index_to_element(index, |index| D::direction_from_index(index)))
    }

    fn number_of_directions() -> usize {
        D::number_of_directions() + 2
    }
}
