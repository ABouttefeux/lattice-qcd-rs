//! Contains Iterator over lattice and direction elements. All implementing
//! [`Iterator`], [`DoubleEndedIterator`], [`rayon::iter::ParallelIterator`],
//! [`rayon::iter::IndexedParallelIterator`], [`ExactSizeIterator`] and
//! [`std::iter::FusedIterator`].

// TODO reduce identical code using private traits

mod direction;
mod link_canonical;
mod point;

use std::fmt::{self, Display};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

pub use self::direction::IteratorDirection;
pub use self::link_canonical::IteratorLatticeLinkCanonical;
pub use self::point::IteratorLatticePoint;

/// Enum for internal use of iterator. It store the previous element returned by `next`
#[allow(clippy::exhaustive_enums)]
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum IteratorElement<T> {
    /// First element of the iterator
    FirstElement,
    /// An element of the iterator
    Element(T),
    /// The Iterator is exhausted
    LastElement,
}

impl<T> IteratorElement<T> {
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

    #[inline]
    #[must_use]
    fn index_to_element<F: FnOnce(usize) -> Option<T>>(index: usize, closure: F) -> Self {
        if index == 0 {
            Self::FirstElement
        } else {
            closure(index - 1).map_or(Self::LastElement, Self::Element)
        }
    }
}

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

/// Returns [`Some`] in the case of [`IteratorElement::Element`], [`None`] otherwise
impl<T> From<IteratorElement<T>> for Option<T> {
    #[inline]
    fn from(element: IteratorElement<T>) -> Self {
        match element {
            IteratorElement::Element(el) => Some(el),
            IteratorElement::LastElement | IteratorElement::FirstElement => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::IteratorElement;

    #[test]
    fn iterator_element() {
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
}
