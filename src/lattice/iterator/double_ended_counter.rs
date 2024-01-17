//! Contains [`DoubleEndedCounter`]

//---------------------------------------
// uses

use std::{
    fmt::{self, Display},
    iter::FusedIterator,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::{Getter, Sealed};

use super::{IteratorElement, RandomAccessIterator, Split};
use crate::lattice::direction::DirectionIndexing;

//---------------------------------------
// struct definition

/// An iterator that track the front and the back in order to be able to implement
/// [`DoubleEndedIterator`].
///
/// By itself it is not use a lot in the library it is used as a properties and use
/// to track the front and the back. [`Iterator`] traits are not (yet ?) implemented
/// on this type.
#[derive(Sealed, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Getter)]
pub struct DoubleEndedCounter<T> {
    /// Front element of the iterator. The state need to be increased before
    /// being returned by the next [`Iterator::next`] call.
    #[get(Const, Pub)]
    #[get_mut(Pub)]
    pub(super) front: IteratorElement<T>,
    /// End element of the iterator.
    /// It needs to be decreased before the next [`DoubleEndedIterator::next_back`] call.
    #[get(Const, Pub)]
    #[get_mut(Pub)]
    pub(super) end: IteratorElement<T>,
}

impl<T> DoubleEndedCounter<T> {
    /// Create a new [`Self`] with [`IteratorElement::FirstElement`] as `front` and
    /// [`IteratorElement::LastElement`] as `end`
    /// # Example
    /// ```ignore
    /// use lattice_qcd_rs::lattice::iter::{DoubleEndedCounter,IteratorElement};
    /// let counter = DoubleEndedCounter::<()>::new();
    /// assert_eq!(counter.front(), IteratorElement::FirstElement);
    /// assert_eq!(counter.end(), IteratorElement::LastElement);
    /// ```
    /// TODO restrict to valid iter ?
    pub const fn new() -> Self {
        Self {
            front: IteratorElement::FirstElement,
            end: IteratorElement::LastElement,
        }
    }

    /// Create a new self with a given `front` and `end` element
    pub(super) const fn new_with_front_end(
        front: IteratorElement<T>,
        end: IteratorElement<T>,
    ) -> Self {
        Self { front, end }
    }

    // possible with_first, with_last
}

//---------------------------------------
// common traits

/// It is [`Self::new`],
impl<T> Default for DoubleEndedCounter<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Display> Display for DoubleEndedCounter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "front: {}, end: {}", self.front(), self.end())
    }
}

//---------------------------------------
// conversion

impl<T> From<DoubleEndedCounter<T>> for [IteratorElement<T>; 2] {
    #[inline]
    fn from(value: DoubleEndedCounter<T>) -> Self {
        [value.front, value.end]
    }
}

impl<T> From<DoubleEndedCounter<T>> for (IteratorElement<T>, IteratorElement<T>) {
    #[inline]
    fn from(value: DoubleEndedCounter<T>) -> Self {
        (value.front, value.end)
    }
}

//---------------------------------------
// impl of RandomAccessIterator

impl<D: DirectionIndexing> RandomAccessIterator for DoubleEndedCounter<D> {
    type Item = D;

    fn iter_len(&self) -> usize {
        // this time it is end - front because we use index and not length
        self.end()
            .direction_to_index()
            .saturating_sub(self.front().direction_to_index())
    }

    fn increase_front_element_by(&self, advance_by: usize) -> IteratorElement<Self::Item> {
        let index = match self.front() {
            IteratorElement::FirstElement => 0,
            IteratorElement::Element(ref element) => element.direction_to_index() + 1,
            IteratorElement::LastElement => {
                // early return
                return IteratorElement::LastElement;
            }
        };

        let new_index = index + advance_by;
        IteratorElement::index_to_element(new_index, |index| {
            Self::Item::direction_from_index(index)
        })
    }

    fn decrease_end_element_by(&self, back_by: usize) -> IteratorElement<Self::Item> {
        let index = match self.end() {
            IteratorElement::FirstElement => {
                // early return
                return IteratorElement::FirstElement;
            }
            IteratorElement::Element(ref element) => element.direction_to_index() + 1,
            IteratorElement::LastElement => Self::Item::number_of_directions() + 1,
        };

        let new_index = index.saturating_sub(back_by);
        IteratorElement::index_to_element(new_index, |index| {
            Self::Item::direction_from_index(index)
        })
    }
}

impl<I> Split for DoubleEndedCounter<I>
where
    Self: RandomAccessIterator<Item = I>,
    I: Clone,
{
    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let splinting = self.increase_front_element_by(index);
        (
            Self::new_with_front_end(self.front, splinting.clone()),
            Self::new_with_front_end(splinting, self.end),
        )
    }
}

//---------------------------------------
// impl of Iterator traits

/// TODO DOC
impl<T> Iterator for DoubleEndedCounter<T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.nth(0)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.iter_len();
        (size, Some(size))
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.iter_len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.nth(self.iter_len().saturating_sub(1))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.iter_len();
        if len <= n {
            if len != 0 {
                // we need to change the state of the iterator other wise it could
                // produce element we should have otherwise skipped.
                //*self.front_mut() = self.end().clone();
                *self.front_mut() = IteratorElement::LastElement;
            }
            return None;
        }
        let next_element = self.increase_front_element_by(n + 1);
        *self.front_mut() = next_element.clone();
        next_element.into()
    }

    #[inline]
    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        self.last()
    }

    #[inline]
    fn min(mut self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        self.next()
    }
}

/// TODO DOC
impl<T> DoubleEndedIterator for DoubleEndedCounter<T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.iter_len();
        if len <= n {
            if len != 0 {
                // we need to change the state of the iterator other wise it could
                // produce element we should have otherwise skipped.
                //*self.end_mut() = self.front().clone();
                *self.end_mut() = IteratorElement::FirstElement;
            }
            return None;
        }
        let previous_element = self.decrease_end_element_by(n + 1);
        *self.end_mut() = previous_element.clone();
        previous_element.into()
    }
}

impl<T> FusedIterator for DoubleEndedCounter<T> where Self: RandomAccessIterator + Iterator {}

impl<T> ExactSizeIterator for DoubleEndedCounter<T>
where
    Self: RandomAccessIterator + Iterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter_len()
    }
}
