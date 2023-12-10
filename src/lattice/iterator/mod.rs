//! Contains Iterator over lattice and direction elements. All implementing
//! [`Iterator`], [`DoubleEndedIterator`], [`rayon::iter::ParallelIterator`],
//! [`rayon::iter::IndexedParallelIterator`], [`ExactSizeIterator`] and
//! [`std::iter::FusedIterator`].

// TODO reduce identical code using private traits

// TODO
// - unify direction behavior (closure)
// - bench
// - possibly optimize next and next_back operation
// - bound to the struct declaration
// - more doc
// - optimize other part of the code

mod direction;
mod link_canonical;
mod point;

use std::{
    fmt::{self, Display},
    iter::FusedIterator,
    mem,
};

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::{Getter, Sealed};

pub use self::direction::IteratorDirection;
pub use self::link_canonical::{IteratorLatticeLinkCanonical, ParIterLatticeLinkCanonical};
pub use self::point::{IteratorLatticePoint, ParIterLatticePoint};
use super::{IndexToElement, LatticeCyclic, LatticeElementToIndex, NumberOfLatticeElement};
use crate::private::Sealed;

/// Trait for generic implementation of [`Iterator`] for implementor of this trait.
///
/// It has a notion of dimension as it is link to the notion of lattice element.
/// And a lattice takes a dimension.
pub trait RandomAccessIterator: Sealed {
    /// Type of element return by the iterator
    type Item;

    /// Returns the number of elements left in the iterator.
    #[must_use]
    fn iter_len(&self) -> usize;

    /// Increase the given front element by the given number and return the result
    /// without modifying the iterator.
    #[must_use]
    fn increase_front_element_by(&self, advance_by: usize) -> IteratorElement<Self::Item>;

    /// Decrease the given end element by the given number and return the result
    /// without modifying the iterator.
    #[must_use]
    fn decrease_end_element_by(&self, back_by: usize) -> IteratorElement<Self::Item>;

    // /// Increase the given front element by the given number and modify the iterator.
    // fn increase_front_by(&mut self, advance_by: usize) {
    //     *self.as_mut().front_mut() = self.increase_front_element_by(advance_by);
    // }

    // /// Decrease the end element by a given number and modify the iterator.
    // fn decrease_end_by(&mut self, back_by: usize) {
    //     *self.as_mut().end_mut() = self.decrease_end_element_by(back_by);
    // }
}

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
            Self::Element(element) => element.to_index(lattice) + 1,
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

/// An iterator that track the front and the back in order to be able to implement
/// [`DoubleEndedIterator`].
///
/// By itself it is not use a lot in the library it is used as a properties and use
/// to track the front and the back. [`Iterator`] traits are not (yet ?) implemented
/// on this type.
#[derive(Getter, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct DoubleEndedCounter<T> {
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
    /// # Example
    /// ```ignore
    /// use lattice_qcd_rs::lattice::iter::{DoubleEndedCounter,IteratorElement};
    /// let counter = DoubleEndedCounter::<()>::new();
    /// assert_eq!(counter.front(), IteratorElement::FirstElement);
    /// assert_eq!(counter.end(), IteratorElement::LastElement);
    /// ```
    pub const fn new() -> Self {
        Self {
            front: IteratorElement::FirstElement,
            end: IteratorElement::LastElement,
        }
    }

    // possible with_first, with_last
}

/// It is [`Self::new`],
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

/// Iterator over a lattice. This is a way to generate fast a ensemble of predetermine
/// element without the need of being allocated for big collection. It is also possible
/// to use a parallel version of the iterator (as explained below).
///
/// This struct is use for generically implementing [`Iterator`], [`DoubleEndedIterator`]
/// [`ExactSizeIterator`] and [`FusedIterator`].
///
/// We can also easily transform to a parallel iterator [`LatticeParIter`] using
/// [`Self::par_iter`], [`Self::as_par_iter`] or [`Self::as_par_iter_mut`].
///
/// It is only used
/// for `T` = [`crate::lattice::LatticePoint`] (see [`IteratorLatticePoint`]) and
/// for `T` = [`crate::lattice::LatticeLinkCanonical`] (see [`IteratorLatticeLinkCanonical`]).
/// (Public) constructors only exist for these possibilities.
///
/// The iterators can be created from [`crate::lattice::LatticeCyclic::get_points`] and
/// [`crate::lattice::LatticeCyclic::get_links`].
///
/// # Example
/// TODO
/// ```
/// todo!()
/// ```
// TODO bound explanation
#[derive(Sealed, Getter, Debug, Clone, PartialEq)]
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
    ///
    /// This method is implemented only for T = [`crate::lattice::LatticePoint`]
    /// or [`crate::lattice::LatticeLinkCanonical`].
    ///
    /// # Example
    /// TODO
    /// ```
    /// use lattice_qcd_rs::lattice::{LatticeCyclic, LatticeIterator, LatticePoint};
    /// use nalgebra::Vector4;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 3)?;
    /// let mut iter = LatticeIterator::new(&lattice);
    ///
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(LatticePoint::new(Vector4::new(0, 0, 0, 0)))
    /// );
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(LatticePoint::new(Vector4::new(1, 0, 0, 0)))
    /// );
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(LatticePoint::new(Vector4::new(2, 0, 0, 0)))
    /// );
    /// assert_eq!(
    ///     iter.next(),
    ///     Some(LatticePoint::new(Vector4::new(0, 1, 0, 0)))
    /// );
    /// // nth forward
    /// assert_eq!(
    ///     iter.nth(5),
    ///     Some(LatticePoint::new(Vector4::new(0, 0, 1, 0)))
    /// );
    ///
    /// // the iterator is double ended so we can poll back
    /// assert_eq!(
    ///     iter.next_back(),
    ///     Some(LatticePoint::new(Vector4::new(2, 2, 2, 2)))
    /// );
    /// assert_eq!(
    ///     iter.next_back(),
    ///     Some(LatticePoint::new(Vector4::new(1, 2, 2, 2)))
    /// );
    /// assert_eq!(
    ///     iter.next_back(),
    ///     Some(LatticePoint::new(Vector4::new(0, 2, 2, 2)))
    /// );
    /// // we can also use `nth_back()`
    /// assert_eq!(
    ///     iter.nth_back(8),
    ///     Some(LatticePoint::new(Vector4::new(0, 2, 1, 2)))
    /// );
    /// # Ok(())
    /// # }
    /// ```
    // TODO
    #[must_use]
    #[inline]
    pub const fn new(lattice: &'a LatticeCyclic<D>) -> Self
    where
        Self: RandomAccessIterator<Item = T>,
        T: Clone,
    {
        Self {
            lattice,
            counter: DoubleEndedCounter::new(),
        }
    }

    /// Get the front element tracker.
    #[must_use]
    #[inline]
    const fn front(&self) -> &IteratorElement<T> {
        self.counter().front()
    }

    /// Get the end element tracker.
    #[must_use]
    #[inline]
    const fn end(&self) -> &IteratorElement<T> {
        self.counter().end()
    }

    /// Get a mutable reference on the front element tracker.
    #[must_use]
    #[inline]
    fn front_mut(&mut self) -> &mut IteratorElement<T> {
        self.counter_mut().front_mut()
    }

    /// Get a mutable reference on the end element tracker.
    #[must_use]
    #[inline]
    fn end_mut(&mut self) -> &mut IteratorElement<T> {
        self.counter_mut().end_mut()
    }

    /// create a new iterator. The first [`LatticeIterator::next()`] will return `first_el`.
    ///
    /// This method is implemented only for T = [`crate::lattice::LatticePoint`]
    /// or [`crate::lattice::LatticeLinkCanonical`].
    ///
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
        Self: RandomAccessIterator<Item = T>,
        T: Clone,
    {
        //TODO / FIXME

        // we can't really decrease the first element so we use some trick.
        let mut s = Self {
            lattice,
            counter: DoubleEndedCounter {
                // we define the front and end reversed.
                // we will swap both value afterward
                front: IteratorElement::LastElement,
                end: IteratorElement::Element(first_el),
            },
        };

        // Then we decrease `end`. Also this is fine we don't verify both end of
        // the iterator so we don't care that the iter should produce None.
        *s.end_mut() = s.decrease_end_element_by(1);
        // we then swap the value to get a properly define iterator
        mem::swap(&mut s.counter.front, &mut s.counter.end);
        s
    }

    /// Convert the iterator into an [`IndexedParallelIterator`].
    #[must_use]
    #[inline]
    pub const fn par_iter(self) -> LatticeParIter<'a, D, T>
    where
        Self: RandomAccessIterator<Item = T>,
    {
        LatticeParIter::new(self)
    }

    /// Take a reference of self and return a reference to an [`IndexedParallelIterator`].
    /// This might not be very useful. Look instead at [`Self::as_par_iter_mut`].
    #[allow(unsafe_code)]
    #[must_use]
    #[inline]
    pub const fn as_par_iter<'b>(&'b self) -> &'b LatticeParIter<'a, D, T> {
        // SAFETY: the representation is transparent and the lifetime is not extended
        unsafe { &*(self as *const Self).cast::<LatticeParIter<'a, D, T>>() }
    }

    /// Take a mutable reference of self and return a mutable reference to an
    /// [`IndexedParallelIterator`]
    #[allow(unsafe_code)]
    #[must_use]
    #[inline]
    pub fn as_par_iter_mut<'b>(&'b mut self) -> &'b mut LatticeParIter<'a, D, T> {
        // SAFETY: the representation is transparent and the lifetime is not extended
        unsafe { &mut *(self as *mut Self).cast::<LatticeParIter<'a, D, T>>() }
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

impl<'a, const D: usize, T> RandomAccessIterator for LatticeIterator<'a, D, T>
where
    T: LatticeElementToIndex<D> + NumberOfLatticeElement<D> + IndexToElement<D>,
{
    type Item = T;

    fn iter_len(&self) -> usize {
        // we use a saturating sub because the front could go further than the back
        // it should not however.
        self.front()
            .size_position(self.lattice())
            .saturating_sub(self.end().size_position(self.lattice()))
    }

    fn increase_front_element_by(&self, advance_by: usize) -> IteratorElement<Self::Item> {
        let index = match self.front() {
            IteratorElement::FirstElement => 0,
            IteratorElement::Element(ref element) => element.to_index(self.lattice()) + 1,
            IteratorElement::LastElement => {
                // early return
                return IteratorElement::LastElement;
            }
        };

        let new_index = index + advance_by;
        IteratorElement::index_to_element(new_index, |index| {
            Self::Item::index_to_element(self.lattice(), index)
        })
    }

    fn decrease_end_element_by(&self, back_by: usize) -> IteratorElement<Self::Item> {
        let index = match self.end() {
            IteratorElement::FirstElement => {
                // early return
                return IteratorElement::FirstElement;
            }
            IteratorElement::Element(ref element) => element.to_index(self.lattice()) + 1,
            IteratorElement::LastElement => self.lattice().number_of_points() + 1,
        };

        let new_index = index.saturating_sub(back_by);
        IteratorElement::index_to_element(new_index, |index| {
            Self::Item::index_to_element(self.lattice(), index)
        })
    }
}

/// TODO DOC
impl<'a, const D: usize, T> Iterator for LatticeIterator<'a, D, T>
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
                *self.front_mut() = self.end().clone();
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
impl<'a, const D: usize, T> DoubleEndedIterator for LatticeIterator<'a, D, T>
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
                *self.end_mut() = self.front().clone();
            }
            return None;
        }
        let previous_element = self.decrease_end_element_by(n + 1);
        *self.end_mut() = previous_element.clone();
        previous_element.into()
    }
}

impl<'a, const D: usize, T> FusedIterator for LatticeIterator<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
    T: Clone,
{
}

impl<'a, const D: usize, T> ExactSizeIterator for LatticeIterator<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
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
        value.into_iterator()
    }
}

impl<'a, const D: usize, T> From<LatticeProducer<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn from(value: LatticeProducer<'a, D, T>) -> Self {
        value.into_iterator()
    }
}

impl<'a, const D: usize, T> IntoParallelIterator for LatticeIterator<'a, D, T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone + Send,
{
    type Iter = LatticeParIter<'a, D, T>;
    type Item = <Self::Iter as ParallelIterator>::Item;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        self.par_iter()
    }
}

// cspell: ignore repr
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
struct LatticeProducer<'a, const D: usize, T>(LatticeIterator<'a, D, T>);

impl<'a, const D: usize, T> LatticeProducer<'a, D, T> {
    /// Convert self into a [`LatticeIterator`]
    #[inline]
    #[must_use]
    fn into_iterator(self) -> LatticeIterator<'a, D, T> {
        self.0
    }

    /// Convert as a reference of [`LatticeIterator`]
    #[inline]
    #[must_use]
    const fn as_iter(&self) -> &LatticeIterator<'a, D, T> {
        &self.0
    }

    /// Convert as a mutable reference of [`LatticeIterator`]
    #[inline]
    #[must_use]
    fn as_iter_mut(&mut self) -> &mut LatticeIterator<'a, D, T> {
        &mut self.0
    }
}

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
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
    T: Clone + Send,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LatticeIterator<'a, D, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_iterator()
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let splinting = self.as_iter().increase_front_element_by(index);
        (
            Self(Self::IntoIter {
                lattice: self.0.lattice,
                counter: DoubleEndedCounter {
                    front: self.0.counter.front,
                    end: splinting.clone(),
                },
            }),
            Self(Self::IntoIter {
                lattice: self.0.lattice,
                counter: DoubleEndedCounter {
                    front: splinting,
                    end: self.0.counter.end,
                },
            }),
        )
    }
}

impl<'a, const D: usize, T> AsRef<LatticeIterator<'a, D, T>> for LatticeProducer<'a, D, T> {
    #[inline]
    fn as_ref(&self) -> &LatticeIterator<'a, D, T> {
        self.as_iter()
    }
}

impl<'a, const D: usize, T> AsMut<LatticeIterator<'a, D, T>> for LatticeProducer<'a, D, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut LatticeIterator<'a, D, T> {
        self.as_iter_mut()
    }
}

impl<'a, const D: usize, T> IntoIterator for LatticeProducer<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
    T: Clone,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LatticeIterator<'a, D, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_iterator()
    }
}

/// [`rayon::iter::ParallelIterator`] and [`rayon::iter::IndexedParallelIterator`]
/// over a lattice. It is only used
/// for `T` = [`crate::lattice::LatticePoint`] (see [`ParIterLatticePoint`]) and
/// for `T` = [`crate::lattice::LatticeLinkCanonical`] (see [`ParIterLatticeLinkCanonical`]).
/// (Public) constructors only exist for these possibilities using [`LatticeIterator::par_iter`].
///
/// It has a transparent representation containing a single field [`LatticeIterator`] allowing
/// transmutation between the two. Though other crate should not rely on this representation.
/// TODO more doc
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeParIter<'a, const D: usize, T>(LatticeIterator<'a, D, T>);

impl<'a, const D: usize, T> LatticeParIter<'a, D, T> {
    // where
    //LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    //T: Clone + Send,

    /// create a new self with `iter` as the wrapped value
    const fn new(iter: LatticeIterator<'a, D, T>) -> Self {
        Self(iter)
    }

    /// Convert the parallel iterator into an [`Iterator`]
    #[must_use]
    #[inline]
    pub fn into_iterator(self) -> LatticeIterator<'a, D, T> {
        self.0
    }

    /// Take a reference of self and return a reference to an [`Iterator`].
    /// This might not be very useful look instead at [`Self::as_iter_mut`]
    #[must_use]
    #[inline]
    pub const fn as_iter(&self) -> &LatticeIterator<'a, D, T> {
        &self.0
    }

    /// Take a mutable reference of self and return a mutable reference to an [`Iterator`].
    #[must_use]
    #[inline]
    pub fn as_iter_mut(&mut self) -> &mut LatticeIterator<'a, D, T> {
        &mut self.0
    }
}

impl<'a, const D: usize, T> ParallelIterator for LatticeParIter<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
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
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
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
        Self::new(value)
    }
}

impl<'a, const D: usize, T> From<LatticeProducer<'a, D, T>> for LatticeParIter<'a, D, T> {
    #[inline]
    fn from(value: LatticeProducer<'a, D, T>) -> Self {
        Self(LatticeIterator::from(value))
    }
}

impl<'a, const D: usize, T> AsRef<LatticeIterator<'a, D, T>> for LatticeParIter<'a, D, T> {
    #[inline]
    fn as_ref(&self) -> &LatticeIterator<'a, D, T> {
        self.as_iter()
    }
}

impl<'a, const D: usize, T> AsMut<LatticeIterator<'a, D, T>> for LatticeParIter<'a, D, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut LatticeIterator<'a, D, T> {
        self.as_iter_mut()
    }
}

impl<'a, const D: usize, T> AsRef<LatticeParIter<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn as_ref(&self) -> &LatticeParIter<'a, D, T> {
        self.as_par_iter()
    }
}

impl<'a, const D: usize, T> AsMut<LatticeParIter<'a, D, T>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut LatticeParIter<'a, D, T> {
        self.as_par_iter_mut()
    }
}

impl<'a, const D: usize, T> IntoIterator for LatticeParIter<'a, D, T>
where
    LatticeIterator<'a, D, T>: RandomAccessIterator<Item = T>,
    T: Clone,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = LatticeIterator<'a, D, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_iterator()
    }
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use super::{
        DoubleEndedCounter, IteratorElement, IteratorLatticeLinkCanonical, IteratorLatticePoint,
    };
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

    #[test]
    fn double_ended_counter_default() {
        assert_eq!(
            DoubleEndedCounter::<()>::default(),
            DoubleEndedCounter::<()>::new()
        );
        assert_eq!(
            DoubleEndedCounter::<LatticeLinkCanonical<4>>::default(),
            DoubleEndedCounter::<LatticeLinkCanonical<4>>::new()
        );
        let counter = DoubleEndedCounter::<()>::new();
        assert_eq!(counter.front(), &IteratorElement::FirstElement);
        assert_eq!(counter.end(), &IteratorElement::LastElement);
    }
}
