//! Contains [`LatticeIterator`]

//---------------------------------------
// uses

use std::{
    fmt::{self, Display},
    iter::FusedIterator,
    mem,
};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use utils_lib::{Getter, Sealed};

use super::{DoubleEndedCounter, IteratorElement, ParIter, RandomAccessIterator, Split};
use crate::lattice::{
    IndexToElement, LatticeCyclic, LatticeElementToIndex, NumberOfLatticeElement,
};

/// Iterator over a lattice. This is a way to generate fast a ensemble of predetermine
/// element without the need of being allocated for big collection. It is also possible
/// to use a parallel version of the iterator (as explained below).
///
/// This struct is use for generically implementing [`Iterator`], [`DoubleEndedIterator`]
/// [`ExactSizeIterator`] and [`FusedIterator`].
///
/// We can also easily transform to a parallel iterator [`ParIter`] using
/// [`Self::par_iter`], [`Self::as_par_iter`] or [`Self::as_par_iter_mut`].
///
/// It is only used
/// for `T` = [`crate::lattice::LatticePoint`] (see [`super::IteratorLatticePoint`]) and
/// for `T` = [`crate::lattice::LatticeLinkCanonical`] (see [`super::IteratorLatticeLinkCanonical`]).
/// (Public) constructors only exist for these possibilities.
///
/// The iterators can be created from [`crate::lattice::LatticeCyclic::get_points`] and
/// [`crate::lattice::LatticeCyclic::get_links`].
///
/// # Example
/// TODO
/// ```
/// use lattice_qcd_rs::lattice::{LatticeCyclic, LatticeIterator, LatticePoint};
/// use rayon::prelude::*;
/// # use lattice_qcd_rs::error::ImplementationError;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let lattice = LatticeCyclic::<4>::new(1_f64, 10)?;
/// let iter = LatticeIterator::<'_, 4, LatticePoint<4>>::new(&lattice);
/// # let long_op = |p| {};
/// let vec = iter
///     .par_iter()
///     .map(|point| long_op(point))
///     .collect::<Vec<_>>();
/// # Ok(())
/// # }
/// ```
// TODO bound explanation
#[derive(Sealed, Debug, Clone, PartialEq, Getter)]
pub struct LatticeIterator<'a, const D: usize, T> {
    /// Reference to the lattice.
    #[get(Const, copy)]
    lattice: &'a LatticeCyclic<D>,

    /// Double ended counter using [`IteratorElement`].
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
        Self::new_with_counter(lattice, DoubleEndedCounter::new())
    }

    /// Create a new [`Self`] with the given [`DoubleEndedCounter`]
    #[must_use]
    #[inline]
    pub(super) const fn new_with_counter(
        lattice: &'a LatticeCyclic<D>,
        counter: DoubleEndedCounter<T>,
    ) -> Self {
        Self { lattice, counter }
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
            counter: DoubleEndedCounter::new_with_front_end(
                // we define the front and end reversed.
                // we will swap both value afterward
                IteratorElement::LastElement,
                IteratorElement::Element(first_el),
            ),
        };

        // Then we decrease `end`. Also this is fine we don't verify both end of
        // the iterator so we don't care that the iter should produce None.
        *s.end_mut() = s.decrease_end_element_by(1);
        // we then swap the value to get a properly define iterator
        mem::swap(&mut s.counter.front, &mut s.counter.end);
        s
    }

    /// Convert the iterator into an [`rayon::iter::IndexedParallelIterator`].
    #[must_use]
    #[inline]
    pub const fn par_iter(self) -> ParIter<Self>
    where
        Self: RandomAccessIterator<Item = T>,
    {
        ParIter::new(self)
    }

    /// Take a reference of self and return a reference to an [`rayon::iter::IndexedParallelIterator`].
    /// This might not be very useful. Look instead at [`Self::as_par_iter_mut`].
    #[allow(unsafe_code)]
    #[allow(clippy::needless_lifetimes)] // I want to be explicit
    #[must_use]
    #[inline]
    pub const fn as_par_iter<'b>(&'b self) -> &'b ParIter<Self> {
        // SAFETY: the representation is transparent and the lifetime is not extended
        unsafe { &*(self as *const Self).cast::<ParIter<Self>>() }
    }

    /// Take a mutable reference of self and return a mutable reference to an
    /// [`rayon::iter::IndexedParallelIterator`]
    #[allow(unsafe_code)]
    #[allow(clippy::needless_lifetimes)] // I want to be explicit
    #[must_use]
    #[inline]
    pub fn as_par_iter_mut<'b>(&'b mut self) -> &'b mut ParIter<Self> {
        // SAFETY: the representation is transparent and the lifetime is not extended
        unsafe { &mut *(self as *mut Self).cast::<ParIter<Self>>() }
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

impl<'a, const D: usize, T> Split for LatticeIterator<'a, D, T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone,
{
    fn split_at(self, index: usize) -> (Self, Self) {
        let splinting = self.increase_front_element_by(index);
        (
            Self::new_with_counter(
                self.lattice,
                DoubleEndedCounter::new_with_front_end(self.counter.front, splinting.clone()),
            ),
            Self::new_with_counter(
                self.lattice,
                DoubleEndedCounter::new_with_front_end(splinting, self.counter.end),
            ),
        )
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
                //*self.front_mut() = self.end().clone();
                *self.front_mut() = IteratorElement::LastElement;
            }
            return None;
        }
        let next_element = self.increase_front_element_by(n + 1);
        *self.front_mut() = next_element;
        // TODO refactor code to save this clone and not have Clone in the bound
        self.front().clone().into()
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
                //*self.end_mut() = self.front().clone();
                *self.end_mut() = IteratorElement::FirstElement;
            }
            return None;
        }
        let previous_element = self.decrease_end_element_by(n + 1);
        *self.end_mut() = previous_element;
        self.end().clone().into()
    }
}

impl<'a, const D: usize, T> FusedIterator for LatticeIterator<'a, D, T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone,
{
}

impl<'a, const D: usize, T> ExactSizeIterator for LatticeIterator<'a, D, T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter_len()
    }
}

impl<'a, const D: usize, T> From<ParIter<LatticeIterator<'a, D, T>>> for LatticeIterator<'a, D, T> {
    #[inline]
    fn from(value: ParIter<LatticeIterator<'a, D, T>>) -> Self {
        value.into_iterator()
    }
}

impl<'a, const D: usize, T> IntoParallelIterator for LatticeIterator<'a, D, T>
where
    Self: RandomAccessIterator<Item = T>,
    T: Clone + Send,
{
    type Iter = ParIter<LatticeIterator<'a, D, T>>;
    type Item = <Self::Iter as ParallelIterator>::Item;

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        self.par_iter()
    }
}
