//! Module for [`IteratorDirection`]
//!
//! # example
//! see [`IteratorDirection`]

#![allow(deprecated)]
use std::iter::FusedIterator;

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{super::Direction, DoubleEndedCounter, IteratorElement};
use crate::lattice::OrientedDirection;

/// Iterator over [`OrientedDirection`].
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::{IteratorOrientedDirection, OrientedDirection, IteratorElement};
/// # use lattice_qcd_rs::error::ImplementationError;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut iter = IteratorOrientedDirection::<4, true>::new();
///
/// let iter_val = iter.next();
/// // debug
/// println!("{iter_val:?}, {:?}", OrientedDirection::<4, true>::new(0));
///
/// assert_eq!(
///     iter_val.ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     OrientedDirection::<4, true>::new(0)
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     OrientedDirection::<4, true>::new(1)
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     OrientedDirection::<4, true>::new(2)
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(
///     iter.next()
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
///     OrientedDirection::<4, true>::new(3)
///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
/// );
/// assert_eq!(iter.next(), None);
/// assert_eq!(iter.next(), None);
/// # Ok(())
/// # }
/// ```
pub type IteratorOrientedDirection<const D: usize, const ORIENTATION: bool> =
    DoubleEndedCounter<OrientedDirection<D, ORIENTATION>>;

/// Iterator over [`Direction`] with the orientation.
///
/// This item is depreciated and used should used [`IteratorOrientedDirection`] instead
/// (whose output can be converted into an [`Direction`]).
///
/// The reason of the depreciation is that I use a new system that use generic trait for
/// the iterators this is an old struct where everything was more or done the same way
/// but manually it means it does not benefit from the optimization and the correction
/// the others iterators benefits from. Also because of the way I count element on the
/// lattice it does not make sense any more.
/// (before direction were counted for d = 2 `(index 0, or: true)`, `(index 1, or: true)` - end of count,
/// negative orientation weren't taken into account. Now they are counted
/// `(axis : 0, or: true)`, `(axis: 0, or false)`, `(axis : 1, or: true)`, `(axis: 1, or false)`).
/// Now it make more sense to iterate over [`OrientedDirection`]
///
///
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
// TODO remove ?
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[deprecated(
    since = "0.3.0",
    note = "use `IteratorOrientedDirection` instead with a map and a conversion"
)]
pub struct IteratorDirection<const D: usize, const IS_POSITIVE_DIRECTION: bool> {
    /// Front element of the iterator. The state need to be increased before
    /// being returned by the next [`Iterator::next`] call.
    front: IteratorElement<Direction<D>>,
    /// End element of the iterator.
    /// It needs to be decreased before the next [`DoubleEndedIterator::next_back`] call.
    end: IteratorElement<Direction<D>>,
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
    #[must_use]
    #[inline]
    pub const fn new(element: Option<Direction<D>>) -> Option<Self> {
        match element {
            None => Self::new_from_element(IteratorElement::FirstElement),
            Some(dir) => Self::new_from_element(IteratorElement::Element(dir)),
        }
    }

    /// create a new iterator. The first call to [`IteratorDirection::next()`] gives the element
    /// just after the one given or the first element if [`IteratorElement::FirstElement`]
    /// is given.
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
    #[must_use]
    #[inline]
    pub const fn new_from_element(element: IteratorElement<Direction<D>>) -> Option<Self> {
        if D == 0 {
            return None;
        }
        if let IteratorElement::Element(ref el) = element {
            if el.is_positive() != IS_POSITIVE_DIRECTION {
                return None;
            }
        }
        Some(Self {
            front: element,
            end: IteratorElement::LastElement,
        })
    }

    /// Computes the number of elements left with the given element as the `front`
    /// of the iterator and the `end` as [`IteratorElement::LastElement`].
    #[inline]
    #[must_use]
    const fn size_position(element: &IteratorElement<Direction<D>>) -> usize {
        match element {
            IteratorElement::FirstElement => D,
            IteratorElement::Element(dir) => D - (dir.index() + 1),
            IteratorElement::LastElement => 0,
        }
    }

    /// returns the number of elements left in the iterator
    #[inline]
    #[must_use]
    const fn iter_len(&self) -> usize {
        // we use a saturating sub because the front could go further than the back
        // it should not however.
        Self::size_position(&self.front).saturating_sub(Self::size_position(&self.end))
    }

    /// Transform a given index in the order of the iterator beginning at
    /// [`IteratorElement::FirstElement`] into a given direction provided this point exits.
    #[inline]
    #[must_use]
    const fn index_to_direction(index: usize) -> IteratorElement<Direction<D>> {
        // TODO
        if index == 0 {
            IteratorElement::FirstElement
        } else if let Some(dir) = Direction::new(index - 1, IS_POSITIVE_DIRECTION) {
            IteratorElement::Element(dir)
        } else {
            IteratorElement::LastElement
        }
    }

    /// Increase an element by a given amount of step.
    #[inline]
    #[must_use]
    const fn increase_element_by(
        front_element: &IteratorElement<Direction<D>>,
        advance_by: usize,
    ) -> IteratorElement<Direction<D>> {
        // note that this is different from D - [`Self::size_position`].
        let front_index = match front_element {
            IteratorElement::FirstElement => 0,
            IteratorElement::Element(dir) => dir.index() + 1,
            IteratorElement::LastElement => {
                // or D + 1 as `advance_by + front_index - 1` should be >= D
                return IteratorElement::LastElement;
            }
        };
        Self::index_to_direction(front_index + advance_by)
    }

    /// Decrease an element by a given amount of step.
    #[inline]
    #[must_use]
    const fn decrease_element_by(
        end_element: &IteratorElement<Direction<D>>,
        back_by: usize,
    ) -> IteratorElement<Direction<D>> {
        let end_index = match end_element {
            IteratorElement::FirstElement => return IteratorElement::FirstElement,
            IteratorElement::Element(dir) => dir.index() + 1,
            IteratorElement::LastElement => D + 1,
        };
        Self::index_to_direction(end_index.saturating_sub(back_by))
    }
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> Iterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    type Item = Direction<D>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // let next_element = match self.front {
        //     IteratorElement::FirstElement => Direction::new(0, IS_POSITIVE_DIRECTION)
        //         .map_or(IteratorElement::LastElement, IteratorElement::Element),
        //     IteratorElement::Element(ref dir) => {
        //         Direction::new(dir.index() + 1, IS_POSITIVE_DIRECTION)
        //             .map_or(IteratorElement::LastElement, IteratorElement::Element)
        //     }
        //     IteratorElement::LastElement => IteratorElement::LastElement,
        // };
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
        // the length could be 0 so we use a saturating sub
        self.nth(self.iter_len().saturating_sub(1))
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.iter_len();
        if len <= n {
            if len != 0 {
                // we need to change the state of the iterator other wise it could
                // produce element we should have otherwise skipped.
                self.front = self.end;
            }
            return None;
        }
        let next_element = Self::increase_element_by(&self.front, n + 1);
        self.front = next_element;
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

    // #[cfg(feature = "experimental")]
    // #[inline]
    // fn is_sorted(self) -> bool
    //     where
    //         Self: Sized,
    //         Self::Item: PartialOrd, {
    //     true
    // }
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> DoubleEndedIterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // let previous_element = match self.end {
        //     IteratorElement::LastElement => {
        //         D.checked_sub(1)
        //             .map_or(IteratorElement::FirstElement, |index| {
        //                 IteratorElement::Element(
        //                     Direction::new(index, IS_POSITIVE_DIRECTION).expect("always exist"),
        //                 )
        //             })
        //     }
        //     IteratorElement::Element(dir) => {
        //         dir.index()
        //             .checked_sub(1)
        //             .map_or(IteratorElement::FirstElement, |index| {
        //                 IteratorElement::Element(
        //                     Direction::new(index, IS_POSITIVE_DIRECTION).expect("always exist"),
        //                 )
        //             })
        //     }
        //     IteratorElement::FirstElement => IteratorElement::FirstElement,
        // };
        self.nth_back(0)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.iter_len();
        if len <= n {
            if len != 0 {
                // we need to change the state of the iterator other wise it could
                // produce element we should have otherwise skipped.
                self.end = self.front;
            }
            return None;
        }
        let previous_element = Self::decrease_element_by(&self.end, n + 1);
        self.end = previous_element;
        previous_element.into()
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

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> ParallelIterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    type Item = <Self as Iterator>::Item;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.iter_len())
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.iter_len()
    }

    #[inline]
    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        <Self as Iterator>::max(self)
    }

    #[inline]
    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        <Self as Iterator>::min(self)
    }
}

impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> IndexedParallelIterator
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    #[inline]
    fn len(&self) -> usize {
        self.iter_len()
    }

    #[inline]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    #[inline]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

// #[doc(hidden)]
impl<const D: usize, const IS_POSITIVE_DIRECTION: bool> Producer
    for IteratorDirection<D, IS_POSITIVE_DIRECTION>
{
    type Item = <Self as Iterator>::Item;
    type IntoIter = Self;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        // TODO test
        let splinting = Self::increase_element_by(&self.front, index);
        (
            Self {
                front: self.front,
                end: splinting,
            },
            Self {
                front: splinting,
                end: self.end,
            },
        )
    }
}

#[cfg(test)]
mod test {
    use super::IteratorDirection;
    use crate::{error::ImplementationError, lattice::Direction};

    fn test_iter<const N: usize, const POSITIVE: bool>() -> Result<(), ImplementationError> {
        let mut iterator = IteratorDirection::<N, POSITIVE>::new(None)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;

        for i in 0..N {
            assert_eq!(iterator.size_hint(), (N - i, Some(N - i)));
            assert_eq!(iterator.next(), Direction::new(i, POSITIVE));
        }
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert!(iterator.next().is_none());
        assert!(iterator.next().is_none());

        Ok(())
    }

    #[test]
    fn test() -> Result<(), ImplementationError> {
        let mut iterator = IteratorDirection::<2, true>::new(None)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq!(iterator.size_hint(), (2, Some(2)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert!(iterator.next().is_none());
        assert!(iterator.next().is_none());

        test_iter::<4, false>()?;
        test_iter::<10, true>()?;
        test_iter::<20, true>()?;

        Ok(())
    }
}
