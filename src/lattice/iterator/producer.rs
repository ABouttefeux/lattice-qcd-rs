//! Contains [`Prod`]

//---------------------------------------
// uses

use rayon::iter::plumbing::Producer;

use super::{ParIter, RandomAccessIterator, Split};

//---------------------------------------
// struct definition

/// [`rayon::iter::plumbing::Producer`] for the [`rayon::iter::IndexedParallelIterator`]
/// [`super::ParIter`] based on the [`DoubleEndedIterator`] [`Prod`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Prod<T>(T);

//---------------------------------------
// impl block

impl<T> Prod<T> {
    /// Create a new [`Self`] with the given value
    const fn new(iter: T) -> Self {
        Self(iter)
    }

    /// Convert self into a [`Prod`]
    #[inline]
    #[must_use]
    pub fn into_iterator(self) -> T {
        self.0
    }

    /// Convert as a reference of [`Prod`]
    #[inline]
    #[must_use]
    const fn as_iter(&self) -> &T {
        &self.0
    }

    /// Convert as a mutable reference of [`Prod`]
    #[allow(clippy::iter_not_returning_iterator)] // yes in some cases (see impl of IntoIterator)
    #[inline]
    #[must_use]
    fn iter_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

//---------------------------------------
// Producer trait

impl<T, I> Producer for Prod<T>
where
    T: RandomAccessIterator<Item = I>
        + Split
        + Iterator<Item = I>
        + ExactSizeIterator
        + DoubleEndedIterator
        + Send,
    I: Clone + Send,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = T;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_iterator()
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let split = self.into_iterator().split_at(index);
        (Self::new(split.0), Self::new(split.1))
    }
}

// impl<T, I> Producer for Prod<T>
// where
//     T: RandomAccessIterator<Item = I>
//         + Iterator<Item = I>
//         + ExactSizeIterator
//         + DoubleEndedIterator
//         + Send,
//     I: Clone + Send,
// {
//     type Item = <Self::IntoIter as Iterator>::Item;
//     type IntoIter = T;

//     #[inline]
//     fn into_iter(self) -> Self::IntoIter {
//         self.into_iterator()
//     }

//     #[inline]
//     fn split_at(self, index: usize) -> (Self, Self) {
//         let splinting = self.as_iter().increase_front_element_by(index);
//         (
//             Self(Self::IntoIter {
//                 lattice: self.0.lattice,
//                 counter: DoubleEndedCounter::new_with_front_end(
//                     self.0.counter.front,
//                     splinting.clone(),
//                 ),
//             }),
//             Self(Self::IntoIter {
//                 lattice: self.0.lattice,
//                 counter: DoubleEndedCounter::new_with_front_end(splinting, self.0.counter.end),
//             }),
//         )
//     }
// }

//---------------------------------------
// IntoIterator traits

impl<T> IntoIterator for Prod<T>
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

impl<'a, T> IntoIterator for &'a mut Prod<T>
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
// conversion trait

impl<T> From<T> for Prod<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> From<ParIter<T>> for Prod<T> {
    #[inline]
    fn from(value: ParIter<T>) -> Self {
        Self(value.into_iterator())
    }
}

impl<T> AsRef<T> for Prod<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.as_iter()
    }
}

impl<T> AsMut<T> for Prod<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.iter_mut()
    }
}
