//! Contains [`ParIter`]

//---------------------------------------
// uses

use std::fmt::{self, Display};

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};

use super::{producer::Prod, LatticeIterator, RandomAccessIterator};

//---------------------------------------
// struct definition

/// [`rayon::iter::ParallelIterator`] and [`rayon::iter::IndexedParallelIterator`]
/// over a lattice. It is only used
/// for `T` = [`crate::lattice::LatticePoint`] (see [`super::ParIterLatticePoint`]) and
/// for `T` = [`crate::lattice::LatticeLinkCanonical`] (see [`super::ParIterLatticeLinkCanonical`]).
/// (Public) constructors only exist for these possibilities using [`LatticeIterator::par_iter`].
///
/// It has a transparent representation containing a single field [`LatticeIterator`] allowing
/// transmutation between the two. Though other crate should not rely on this representation.
/// TODO more doc
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParIter<T>(T);

//---------------------------------------
// impl block

impl<T> ParIter<T> {
    // where
    //LatticeIterator<'a, D, T>: RandomAccessIterator<D, Item = T>,
    //T: Clone + Send,

    /// create a new self with `iter` as the wrapped value
    pub(super) const fn new(iter: T) -> Self {
        Self(iter)
    }

    /// Convert the parallel iterator into an [`Iterator`]
    #[must_use]
    #[inline]
    pub fn into_iterator(self) -> T {
        self.0
    }

    /// Take a reference of self and return a reference to an [`Iterator`].
    /// This might not be very useful look instead at [`Self::iter_mut`]
    #[must_use]
    #[inline]
    pub const fn as_iter(&self) -> &T {
        &self.0
    }

    /// Take a mutable reference of self and return a mutable reference to an [`Iterator`].
    #[allow(clippy::iter_not_returning_iterator)] // yes in some cases (see impl of IntoIterator)
    #[must_use]
    #[inline]
    pub fn iter_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

//---------------------------------------
// common trait

impl<T: Display> Display for ParIter<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_iter())
    }
}

//---------------------------------------
// Parallel iterator traits

impl<T, I> IndexedParallelIterator for ParIter<T>
where
    T: RandomAccessIterator<Item = I>
        + Iterator<Item = I>
        + ExactSizeIterator
        + DoubleEndedIterator
        + Send,
    I: Clone + Send,
    Prod<T>: Producer<Item = I, IntoIter = T>,
{
    #[inline]
    fn len(&self) -> usize {
        self.as_iter().iter_len()
    }

    #[inline]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    #[inline]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(Prod::<T>::from(self))
    }
}

impl<T, I> ParallelIterator for ParIter<T>
where
    T: RandomAccessIterator<Item = I>
        + Iterator<Item = I>
        + ExactSizeIterator
        + DoubleEndedIterator
        + Send,
    I: Clone + Send,
    Prod<T>: Producer<Item = I, IntoIter = T>,
{
    type Item = I;

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
        <T as Iterator>::max(self.into_iterator())
    }

    #[inline]
    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        <T as Iterator>::min(self.into_iterator())
    }
}

//---------------------------------------
// IntoIter traits

impl<T> IntoIterator for ParIter<T>
where
    T: Iterator,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_iterator()
    }
}

impl<'a, T> IntoIterator for &'a mut ParIter<T>
where
    T: Iterator,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = &'a mut T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

//---------------------------------------
// conversion traits

impl<'a, const D: usize, T> From<LatticeIterator<'a, D, T>> for ParIter<LatticeIterator<'a, D, T>> {
    #[inline]
    fn from(value: LatticeIterator<'a, D, T>) -> Self {
        Self::new(value)
    }
}

impl<T> From<Prod<T>> for ParIter<T> {
    #[inline]
    fn from(value: Prod<T>) -> Self {
        Self(value.into_iterator())
    }
}

impl<T> AsRef<T> for ParIter<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.as_iter()
    }
}

impl<T> AsMut<T> for ParIter<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.iter_mut()
    }
}

impl<'a, const D: usize, T> AsRef<ParIter<LatticeIterator<'a, D, T>>>
    for LatticeIterator<'a, D, T>
{
    #[inline]
    fn as_ref(&self) -> &ParIter<LatticeIterator<'a, D, T>> {
        self.as_par_iter()
    }
}

impl<'a, const D: usize, T> AsMut<ParIter<LatticeIterator<'a, D, T>>>
    for LatticeIterator<'a, D, T>
{
    #[inline]
    fn as_mut(&mut self) -> &mut ParIter<LatticeIterator<'a, D, T>> {
        self.as_par_iter_mut()
    }
}
