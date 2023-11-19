//! Module for [`IteratorLatticeLinkCanonical`]

use std::iter::FusedIterator;

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};

use super::{
    super::{
        Direction, IteratorDirection, LatticeCyclic, LatticeElementToIndex, LatticeLinkCanonical,
    },
    IteratorElement, IteratorLatticePoint,
};

/// Iterator over [`LatticeLinkCanonical`] associated to a particular [`LatticeCyclic`].
#[derive(Clone, Debug, PartialEq)]
pub struct IteratorLatticeLinkCanonical<'a, const D: usize> {
    /// Ref to the lattice
    lattice: &'a LatticeCyclic<D>,
    /// Front element of the iterator. The state need to be increased before
    /// being returned by the next [`Iterator::next`] call.
    front: IteratorElement<LatticeLinkCanonical<D>>,
    /// End element of the iterator.
    /// It needs to be decreased before the next [`DoubleEndedIterator::next_back`] call.
    end: IteratorElement<LatticeLinkCanonical<D>>,
}

impl<'a, const D: usize> IteratorLatticeLinkCanonical<'a, D> {
    /// create a new iterator. The first [`IteratorLatticeLinkCanonical::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticeLinkCanonical, LatticeCyclic, LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// #
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// #
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let first_el = LatticeLinkCanonical::<4>::new(LatticePoint::from([1, 0, 2, 0]), DirectionEnum::YPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let mut iter = IteratorLatticeLinkCanonical::new(&lattice, first_el);
    /// assert_eq!(iter.next().ok_or(ImplementationError::OptionWithUnexpectedNone)?, first_el);
    /// #
    /// # Ok(())
    /// # }
    /// ```
    // TODO const fn ?
    #[must_use]
    #[inline]
    pub fn new(lattice: &'a LatticeCyclic<D>, first_el: LatticeLinkCanonical<D>) -> Self {
        Self {
            lattice,
            front: Self::decrease_element_by(lattice, &IteratorElement::Element(first_el), 1),
            end: IteratorElement::LastElement,
        }
    }

    /// Computes the number of elements left with the given element as the `front`
    /// of the iterator and the `back` as [`None`].
    #[inline]
    #[must_use]
    fn size_position(
        lattice: &LatticeCyclic<D>,
        link: &IteratorElement<LatticeLinkCanonical<D>>,
    ) -> usize {
        #[allow(clippy::option_if_let_else)] // for possible const function later
        match link {
            IteratorElement::FirstElement => lattice.number_of_canonical_links_space(),
            IteratorElement::Element(link) => {
                lattice.number_of_canonical_links_space() - (link.to_index(lattice) + 1)
            }
            IteratorElement::LastElement => 0,
        }
    }

    /// returns the number of elements left in the iterator
    #[must_use]
    #[inline]
    fn iter_len(&self) -> usize {
        // we use a saturating sub because the front could go further than the back
        // it should not however.
        Self::size_position(self.lattice, &self.front)
            .saturating_sub(Self::size_position(self.lattice, &self.end))
    }

    /// Transform a given index in the order of the iterator beginning at the
    /// [`LatticeLinkCanonical`] zero into a given [`LatticeLinkCanonical`] provided
    /// this link exits in the lattice.
    #[inline]
    #[must_use]
    fn index_to_lattice_link(
        lattice: &LatticeCyclic<D>,
        index: usize,
    ) -> IteratorElement<LatticeLinkCanonical<D>> {
        IteratorElement::index_to_element(index, |index| {
            LatticeLinkCanonical::index_to_canonical_link(lattice, index)
        })
    }

    /// Increase an element by a given amount of step.
    #[inline]
    #[must_use]
    fn increase_element_by(
        lattice: &LatticeCyclic<D>,
        front_element: &IteratorElement<<Self as Iterator>::Item>,
        advance_by: usize,
    ) -> IteratorElement<<Self as Iterator>::Item> {
        let index = match front_element {
            IteratorElement::FirstElement => 0,
            IteratorElement::Element(element) => element.to_index(lattice) + 1,
            IteratorElement::LastElement => return IteratorElement::LastElement,
        };
        let new_index = index + advance_by;
        Self::index_to_lattice_link(lattice, new_index)
    }

    /// Decrease an element by a given amount of step.
    #[inline]
    #[must_use]
    fn decrease_element_by(
        lattice: &LatticeCyclic<D>,
        end_element: &IteratorElement<<Self as Iterator>::Item>,
        back_by: usize,
    ) -> IteratorElement<<Self as Iterator>::Item> {
        let index = match end_element {
            IteratorElement::FirstElement => return IteratorElement::FirstElement,
            IteratorElement::Element(element) => element.to_index(lattice) + 1,
            IteratorElement::LastElement => lattice.number_of_points() + 1,
        };

        let new_index = index.saturating_sub(back_by);
        Self::index_to_lattice_link(lattice, new_index)
    }
}

impl<'a, const D: usize> Iterator for IteratorLatticeLinkCanonical<'a, D> {
    type Item = LatticeLinkCanonical<D>;

    // TODO improve
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // let previous_el = self.front;
        // if let Some(ref mut element) = self.front {
        //     let mut iter_dir = IteratorDirection::<D, true>::new(Some(element.dir))?; // always exist
        //     let new_dir = iter_dir.next();
        //     if let Some(dir) = new_dir {
        //         element.set_dir(dir);
        //     } else {
        //         element.set_dir(Direction::new(0, true)?); // always exist
        //         let mut iter = IteratorLatticePoint::new(self.lattice, *element.pos());
        //         if let Some(array) = iter.nth(1) {
        //             // get the second element
        //             *element.pos_mut() = array;
        //         } else {
        //             self.front = None;
        //             return previous_el;
        //         }
        //     }
        // }
        // previous_el
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
                self.front = self.end;
            }
            return None;
        }
        todo!()
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

impl<'a, const D: usize> DoubleEndedIterator for IteratorLatticeLinkCanonical<'a, D> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        todo!()
    }
}

impl<'a, const D: usize> FusedIterator for IteratorLatticeLinkCanonical<'a, D> {}

impl<'a, const D: usize> ExactSizeIterator for IteratorLatticeLinkCanonical<'a, D> {}

impl<'a, const D: usize> ParallelIterator for IteratorLatticeLinkCanonical<'a, D> {
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

impl<'a, const D: usize> IndexedParallelIterator for IteratorLatticeLinkCanonical<'a, D> {
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

impl<'a, const D: usize> Producer for IteratorLatticeLinkCanonical<'a, D> {
    type Item = <Self as Iterator>::Item;

    type IntoIter = Self;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let splinting = Self::increase_element_by(self.lattice, &self.front, index);
        (
            Self {
                lattice: self.lattice,
                front: self.front,
                end: splinting,
            },
            Self {
                lattice: self.lattice,
                front: splinting,
                end: self.end,
            },
        )
    }
}

#[cfg(test)]
mod test {
    use crate::{error::LatticeInitializationError, lattice::LatticeCyclic};

    #[test]
    fn iterator() -> Result<(), LatticeInitializationError> {
        let l = LatticeCyclic::<2>::new(1_f64, 4)?;
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

        Ok(())
    }
}
