//! Contains Iterator over lattice and direction elements. All implementing
//! [`Iterator`], [`DoubleEndedIterator`], [`rayon::iter::ParallelIterator`],
//! [`rayon::iter::IndexedParallelIterator`], [`ExactSizeIterator`] and
//! [`std::iter::FusedIterator`].

// TODO reduce identical code using private traits

mod direction;
mod link_canonical;
mod point;

use std::{
    fmt::{self, Display},
    iter::FusedIterator,
};

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::Getter;

pub use self::direction::IteratorDirection;
pub use self::link_canonical::IteratorLatticeLinkCanonical;
pub use self::point::IteratorLatticePoint;
use super::{IndexToElement, LatticeCyclic, LatticeElementToIndex, NumberOfLatticeElement};
use crate::private::Sealed;

pub trait RandomAccessIterator<const D: usize>: Sealed {
    type Item;
    // type Iterator: Iterator<Item = &'a Self::Item>;
    // type ParallelIterator: IndexedParallelIterator<Item = &'a Self::Item>;

    /// returns the number of elements left in the iterator
    #[must_use]
    fn iter_len(&self) -> usize;

    #[must_use]
    fn increase_element_by(
        lattice: &LatticeCyclic<D>,
        front_element: &IteratorElement<Self::Item>,
        advance_by: usize,
    ) -> IteratorElement<Self::Item>;

    #[must_use]
    fn decrease_element_by(
        lattice: &LatticeCyclic<D>,
        end_element: &IteratorElement<Self::Item>,
        back_by: usize,
    ) -> IteratorElement<Self::Item>;
}

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

    #[inline]
    #[must_use]
    fn index_to_element<F: FnOnce(usize) -> Option<T>>(index: usize, closure: F) -> Self {
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
    fn size_position<const D: usize>(&self, lattice: &LatticeCyclic<D>) -> usize
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

#[doc(hidden)]
impl<T: Sealed> Sealed for IteratorElement<T> {}

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

impl<const D: usize, T: LatticeElementToIndex<D> + NumberOfLatticeElement<D>>
    LatticeElementToIndex<D> for IteratorElement<T>
{
    #[inline]
    fn to_index(&self, lattice: &LatticeCyclic<D>) -> usize {
        match self {
            Self::FirstElement => 0,
            Self::Element(element) => element.to_index(lattice) + 1,
            Self::LastElement => T::number_of_elements(lattice) + 1,
        }
    }
}

impl<const D: usize, T: LatticeElementToIndex<D> + NumberOfLatticeElement<D>>
    NumberOfLatticeElement<D> for IteratorElement<T>
{
    #[inline]
    fn number_of_elements(lattice: &LatticeCyclic<D>) -> usize {
        T::number_of_elements(lattice) + 2
    }
}

#[derive(Getter, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct DoubleEndedCounter<T> {
    /// Front element of the iterator. The state need to be increased before
    /// being returned by the next [`Iterator::next`] call.
    #[get(Const)]
    #[get_mut]
    front: IteratorElement<T>,
    /// End element of the iterator.
    /// It needs to be decreased before the next [`DoubleEndedIterator::next_back`] call.
    #[get(Const)]
    #[get_mut]
    end: IteratorElement<T>,
}

impl<T> DoubleEndedCounter<T> {
    /// Create a new [`Self`] with [`IteratorElement::FirstElement`] as `front` and
    /// [`IteratorElement::LastElement`] as `end`
    pub const fn new() -> Self {
        Self {
            front: IteratorElement::FirstElement,
            end: IteratorElement::LastElement,
        }
    }
}

impl<T> Default for DoubleEndedCounter<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

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

impl<T: Display> Display for DoubleEndedCounter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "front: {}, end: {}", self.front(), self.end())
    }
}

#[derive(Getter, Debug, Clone, PartialEq)]
pub struct LatticeIterator<'a, const D: usize, T> {
    /// ref to the lattice
    #[get(Const, copy)]
    lattice: &'a LatticeCyclic<D>,

    /// double ended counter using [`IteratorElement`].
    #[get(Const)]
    #[get_mut]
    counter: DoubleEndedCounter<T>,
}

impl<'a, const D: usize, T> LatticeIterator<'a, D, T> {
    /// create a new iterator with a ref to a given lattice, [`IteratorElement::FirstElement`]
    /// as the `front` and [`IteratorElement::LastElement`] as the `end`.
    #[must_use]
    #[inline]
    pub const fn new(lattice: &'a LatticeCyclic<D>) -> Self {
        Self {
            lattice,
            counter: DoubleEndedCounter::new(),
        }
    }

    #[must_use]
    #[inline]
    const fn front(&self) -> &IteratorElement<T> {
        self.counter().front()
    }

    #[must_use]
    #[inline]
    const fn end(&self) -> &IteratorElement<T> {
        self.counter().end()
    }

    #[must_use]
    #[inline]
    fn front_mut(&mut self) -> &mut IteratorElement<T> {
        self.counter_mut().front_mut()
    }

    #[must_use]
    #[inline]
    fn end_mut(&mut self) -> &mut IteratorElement<T> {
        self.counter_mut().end_mut()
    }

    /// create a new iterator. The first [`LatticeIterator::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// use lattice_qcd_rs::error::ImplementationError;
    /// use lattice_qcd_rs::lattice::{
    ///     Direction, DirectionEnum, IteratorLatticeLinkCanonical, IteratorLatticePoint,
    ///     LatticeCyclic, LatticeLinkCanonical, LatticePoint,
    /// };
    /// #
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let lattice = LatticeCyclic::new(1_f64, 4)?;
    /// let first_el = LatticePoint::from([1, 0, 2, 0]);
    /// let mut iter = IteratorLatticePoint::new_with_first_element(&lattice, first_el);
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     first_el
    /// );
    ///
    /// let first_el = LatticeLinkCanonical::<4>::new(
    ///     LatticePoint::from([1, 0, 2, 0]),
    ///     DirectionEnum::YPos.into(),
    /// )
    /// .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let mut iter = IteratorLatticeLinkCanonical::new_with_first_element(&lattice, first_el);
    /// assert_eq!(
    ///     iter.next()
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
    ///     first_el
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn new_with_first_element(lattice: &'a LatticeCyclic<D>, first_el: T) -> Self
    where
        Self: RandomAccessIterator<D, Item = T>,
    {
        Self {
            lattice,
            counter: DoubleEndedCounter {
                front: Self::decrease_element_by(lattice, &IteratorElement::Element(first_el), 1),
                end: IteratorElement::LastElement,
            },
        }
    }

    #[must_use]
    #[inline]
    pub fn par_iter(self) -> LatticeParIter<'a, D, T>
    where
        Self: RandomAccessIterator<D, Item = T>,
    {
        self.into()
    }

    #[must_use]
    #[inline]
    pub fn as_par_iter(&self) -> &LatticeParIter<'a, D, T>
    where
        Self: RandomAccessIterator<D, Item = T>,
    {
        self.as_ref()
    }

    #[must_use]
    #[inline]
    pub fn as_par_iter_mut(&mut self) -> &mut LatticeParIter<'a, D, T>
    where
        Self: RandomAccessIterator<D, Item = T>,
    {
        self.as_mut()
    }
}

// TODO IntoIter trait

/// Simply display the counter
impl<'a, const D: usize, T: Display> Display for LatticeIterator<'a, D, T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.counter())
    }
}

impl<'a, const D: usize, T> Sealed for LatticeIterator<'a, D, T> {}

impl<'a, const D: usize, T> AsRef<DoubleEndedCounter<T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn as_ref(&self) -> &DoubleEndedCounter<T> {
        self.counter()
    }
}

impl<'a, const D: usize, T> AsMut<DoubleEndedCounter<T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut DoubleEndedCounter<T> {
        self.counter_mut()
    }
}

impl<'a, const D: usize, T> AsRef<LatticeCyclic<D>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn as_ref(&self) -> &LatticeCyclic<D> {
        self.lattice()
    }
}

impl<'a, const D: usize, T> From<LatticeIterator<'a, D, T>> for DoubleEndedCounter<T> {
    #[inline]
    fn from(value: LatticeIterator<'a, D, T>) -> Self {
        value.counter
    }
}

impl<'a, const D: usize, T> From<&'a LatticeIterator<'a, D, T>> for &'a LatticeCyclic<D> {
    #[inline]
    fn from(value: &'a LatticeIterator<'a, D, T>) -> Self {
        value.lattice()
    }
}

impl<'a, const D: usize, T> RandomAccessIterator<D> for LatticeIterator<'a, D, T>
where
    T: LatticeElementToIndex<D> + NumberOfLatticeElement<D> + IndexToElement<D>,
{
    type Item = T;
    // type Iterator;
    // type ParallelIterator;

    fn iter_len(&self) -> usize {
        // we use a saturating sub because the front could go further than the back
        // it should not however.
        self.front()
            .size_position(self.lattice())
            .saturating_sub(self.end().size_position(self.lattice()))
    }

    /// Increase an element by a given amount of step.
    fn increase_element_by(
        lattice: &LatticeCyclic<D>,
        front_element: &IteratorElement<Self::Item>,
        advance_by: usize,
    ) -> IteratorElement<Self::Item> {
        let index = match front_element {
            IteratorElement::FirstElement => 0,
            IteratorElement::Element(ref element) => element.to_index(lattice) + 1,
            IteratorElement::LastElement => return IteratorElement::LastElement,
        };
        let new_index = index + advance_by;
        IteratorElement::index_to_element(new_index, |index| {
            Self::Item::index_to_element(lattice, index)
        })
    }

    /// Decrease an element by a given amount of step.
    fn decrease_element_by(
        lattice: &LatticeCyclic<D>,
        end_element: &IteratorElement<Self::Item>,
        back_by: usize,
    ) -> IteratorElement<Self::Item> {
        let index = match end_element {
            IteratorElement::FirstElement => return IteratorElement::FirstElement,
            IteratorElement::Element(ref element) => element.to_index(lattice) + 1,
            IteratorElement::LastElement => lattice.number_of_points() + 1,
        };

        let new_index = index.saturating_sub(back_by);
        IteratorElement::index_to_element(new_index, |index| {
            Self::Item::index_to_element(lattice, index)
        })
    }
}

/// TODO DOC
impl<'a, const D: usize, T> Iterator for LatticeIterator<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
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
                *self.front_mut() = self.end().clone();
            }
            return None;
        }
        let next_element = Self::increase_element_by(self.lattice(), self.front(), n + 1);
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
impl<'a, const D: usize, T> DoubleEndedIterator for LatticeIterator<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
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
                *self.end_mut() = self.front().clone();
            }
            return None;
        }
        let previous_element = Self::decrease_element_by(self.lattice(), self.end(), n + 1);
        *self.end_mut() = previous_element.clone();
        previous_element.into()
    }
}

impl<'a, const D: usize, T> FusedIterator for LatticeIterator<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    T: Clone,
{
}

impl<'a, const D: usize, T> ExactSizeIterator for LatticeIterator<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    T: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter_len()
    }
}

impl<'a, const D: usize, T> From<LatticeParIter<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn from(value: LatticeParIter<'a, D, T>) -> Self {
        value.0
    }
}

impl<'a, const D: usize, T> From<LatticeProducer<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn from(value: LatticeProducer<'a, D, T>) -> Self {
        value.0
    }
}

// cspell: ignore repr
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
struct LatticeProducer<'a, const D: usize, T>(LatticeIterator<'a, D, T>);

impl<'a, const D: usize, T> From<LatticeIterator<'a, D, T>> for LatticeProducer<'a, D, T> {
    #[inline]
    fn from(value: LatticeIterator<'a, D, T>) -> Self {
        Self(value)
    }
}

impl<'a, const D: usize, T> From<LatticeParIter<'a, D, T>> for LatticeProducer<'a, D, T> {
    #[inline]
    fn from(value: LatticeParIter<'a, D, T>) -> Self {
        Self(LatticeIterator::from(value))
    }
}

impl<'a, const D: usize, T> Producer for LatticeProducer<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    T: Clone + Send,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LatticeIterator<'a, D, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into()
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let splinting = Self::IntoIter::increase_element_by(
            self.as_ref().lattice(),
            self.as_ref().front(),
            index,
        );
        (
            Self::IntoIter {
                lattice: self.0.lattice,
                counter: DoubleEndedCounter {
                    front: self.0.counter.front,
                    end: splinting.clone(),
                },
            }
            .into(),
            Self::IntoIter {
                lattice: self.0.lattice,
                counter: DoubleEndedCounter {
                    front: splinting,
                    end: self.0.counter.end,
                },
            }
            .into(),
        )
    }
}

impl<'a, const D: usize, T> AsRef<LatticeIterator<'a, D, T>> for LatticeProducer<'a, D, T> {
    fn as_ref(&self) -> &LatticeIterator<'a, D, T> {
        &self.0
    }
}

impl<'a, const D: usize, T> AsMut<LatticeIterator<'a, D, T>> for LatticeProducer<'a, D, T> {
    fn as_mut(&mut self) -> &mut LatticeIterator<'a, D, T> {
        &mut self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeParIter<'a, const D: usize, T>(LatticeIterator<'a, D, T>);

impl<'a, const D: usize, T> LatticeParIter<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    T: Clone + Send,
{
    #[must_use]
    #[inline]
    pub fn into_iter(self) -> LatticeIterator<'a, D, T>
    where
        LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    {
        self.into()
    }

    #[must_use]
    #[inline]
    pub fn iter(&self) -> &LatticeIterator<'a, D, T>
    where
        LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    {
        self.as_ref()
    }

    #[must_use]
    #[inline]
    pub fn iter_mut(&mut self) -> &mut LatticeIterator<'a, D, T>
    where
        LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    {
        self.as_mut()
    }
}

impl<'a, const D: usize, T> ParallelIterator for LatticeParIter<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    T: Clone + Send,
{
    type Item = T;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.as_ref().iter_len())
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.as_ref().iter_len()
    }

    #[inline]
    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        <LatticeIterator<'a, D, T> as Iterator>::max(self.into())
    }

    #[inline]
    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        <LatticeIterator<'a, D, T> as Iterator>::min(self.into())
    }
}

impl<'a, const D: usize, T> IndexedParallelIterator for LatticeParIter<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    T: Clone + Send,
{
    #[inline]
    fn len(&self) -> usize {
        self.as_ref().iter_len()
    }

    #[inline]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    #[inline]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(LatticeProducer::from(self))
    }
}

impl<'a, const D: usize, T> From<LatticeIterator<'a, D, T>> for LatticeParIter<'a, D, T> {
    #[inline]
    fn from(value: LatticeIterator<'a, D, T>) -> Self {
        Self(value)
    }
}

impl<'a, const D: usize, T> From<LatticeProducer<'a, D, T>> for LatticeParIter<'a, D, T> {
    #[inline]
    fn from(value: LatticeProducer<'a, D, T>) -> Self {
        Self(LatticeIterator::from(value))
    }
}

impl<'a, const D: usize, T> AsRef<LatticeIterator<'a, D, T>> for LatticeParIter<'a, D, T> {
    fn as_ref(&self) -> &LatticeIterator<'a, D, T> {
        &self.0
    }
}

impl<'a, const D: usize, T> AsMut<LatticeIterator<'a, D, T>> for LatticeParIter<'a, D, T> {
    fn as_mut(&mut self) -> &mut LatticeIterator<'a, D, T> {
        &mut self.0
    }
}

impl<'a, const D: usize, T> AsRef<LatticeParIter<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    #[allow(unsafe_code)]
    fn as_ref<'b>(&'b self) -> &'b LatticeParIter<'a, D, T> {
        // SAFETY: the representation is transparent and the lifetime is not extended
        unsafe { &*(self as *const Self).cast::<LatticeParIter<'a, D, T>>() }
    }
}

impl<'a, const D: usize, T> AsMut<LatticeParIter<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    #[allow(unsafe_code)]
    fn as_mut(&mut self) -> &mut LatticeParIter<'a, D, T> {
        // SAFETY: the representation is transparent and the lifetime is not extended
        unsafe { &mut *(self as *mut Self).cast::<LatticeParIter<'a, D, T>>() }
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use super::{IteratorElement, IteratorLatticeLinkCanonical, IteratorLatticePoint};
    use crate::{
        error::ImplementationError,
        lattice::{DirectionEnum, LatticeCyclic, LatticeLinkCanonical, LatticePoint},
    };

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

    #[test]
    fn lattice_iter_new_with_first_el() -> Result<(), Box<dyn Error>> {
        let lattice = LatticeCyclic::new(1_f64, 4)?;
        let first_el = LatticePoint::from([1, 0, 2, 0]);
        let mut iter = IteratorLatticePoint::new_with_first_element(&lattice, first_el);
        assert_eq!(
            iter.next()
                .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
            first_el
        );

        let first_el = LatticeLinkCanonical::<4>::new(
            LatticePoint::from([1, 0, 2, 0]),
            DirectionEnum::YPos.into(),
        )
        .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        let mut iter = IteratorLatticeLinkCanonical::new_with_first_element(&lattice, first_el);
        assert_eq!(
            iter.next()
                .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
            first_el
        );

        Ok(())
    }
}
