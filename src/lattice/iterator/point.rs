//! Module for [`IteratorLatticePoint`]

use std::iter::FusedIterator;

use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};

use super::{
    super::{LatticeCyclic, LatticeElementToIndex, LatticePoint},
    IteratorElement,
};

/// Iterator over [`LatticePoint`]
#[derive(Clone, Debug, PartialEq)]
pub struct IteratorLatticePoint<'a, const D: usize> {
    /// ref to the lattice§
    lattice: &'a LatticeCyclic<D>,
    /// Front element of the iterator. The state need to be increased before
    /// being returned by the next [`Iterator::next`] call.
    front: IteratorElement<LatticePoint<D>>,
    /// end element of the iterator.
    /// It needs to be decreased before the next [`DoubleEndedIterator::next_back`] call.
    end: IteratorElement<LatticePoint<D>>,
}

impl<'a, const D: usize> IteratorLatticePoint<'a, D> {
    /// create a new iterator. The first [`IteratorLatticePoint::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticePoint, LatticeCyclic, LatticePoint, Direction};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::new(1_f64, 4)?;
    /// let first_el = LatticePoint::from([1, 0, 2, 0]);
    /// let mut iter = IteratorLatticePoint::new(&lattice, first_el);
    /// assert_eq!(iter.next().ok_or(ImplementationError::OptionWithUnexpectedNone)?, first_el);
    /// # Ok(())
    /// # }
    /// ```
    // TODO const fn ?
    #[must_use]
    #[inline]
    pub fn new(lattice: &'a LatticeCyclic<D>, first_el: LatticePoint<D>) -> Self {
        Self {
            lattice,
            front: Self::decrease_element_by(lattice, &IteratorElement::Element(first_el), 1),
            end: IteratorElement::LastElement,
        }
    }

    /// Computes the number of elements left with the given element as the `front`
    /// of the iterator and the `back` as [`None`].
    #[must_use]
    #[inline]
    fn size_position(
        lattice: &LatticeCyclic<D>,
        point: &IteratorElement<LatticePoint<D>>,
    ) -> usize {
        #[allow(clippy::option_if_let_else)] // for possible const function later
        match point {
            IteratorElement::FirstElement => lattice.number_of_points(),
            IteratorElement::Element(point) => {
                lattice.number_of_points() - (point.to_index(lattice) + 1)
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
    /// [`LatticePoint`] zero into a given [`LatticePoint`] provided this point
    /// exits in the lattice.
    #[inline]
    #[must_use]
    fn index_to_lattice_point(
        lattice: &LatticeCyclic<D>,
        index: usize,
    ) -> IteratorElement<LatticePoint<D>> {
        IteratorElement::index_to_element(index, |index| {
            LatticePoint::index_to_point(lattice, index)
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
        Self::index_to_lattice_point(lattice, new_index)
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
        Self::index_to_lattice_point(lattice, new_index)
    }
}

impl<'a, const D: usize> Iterator for IteratorLatticePoint<'a, D> {
    type Item = LatticePoint<D>;

    // TODO improve
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // let previous_el = self.front;
        // if let Some(ref mut el) = &mut self.front {
        //     el[0] += 1;
        //     for i in 0..el.len() {
        //         while el[i] >= self.lattice.dim() {
        //             if i < el.len() - 1 {
        //                 // every element except the last one
        //                 el[i + 1] += 1;
        //             } else {
        //                 self.front = None;
        //                 return previous_el;
        //             }
        //             el[i] -= self.lattice.dim();
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
        let next_element = Self::increase_element_by(self.lattice, &self.front, n + 1);
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
}

impl<'a, const D: usize> DoubleEndedIterator for IteratorLatticePoint<'a, D> {
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
                self.end = self.front;
            }
            return None;
        }
        let previous_element = Self::decrease_element_by(self.lattice, &self.end, n + 1);
        self.end = previous_element;
        previous_element.into()
    }
}

impl<'a, const D: usize> FusedIterator for IteratorLatticePoint<'a, D> {}

impl<'a, const D: usize> ExactSizeIterator for IteratorLatticePoint<'a, D> {}

impl<'a, const D: usize> ParallelIterator for IteratorLatticePoint<'a, D> {
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

impl<'a, const D: usize> IndexedParallelIterator for IteratorLatticePoint<'a, D> {
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

impl<'a, const D: usize> Producer for IteratorLatticePoint<'a, D> {
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
    fn test() -> Result<(), LatticeInitializationError> {
        let l = LatticeCyclic::<2>::new(1_f64, 4)?;
        let mut iterator = l.get_points();
        assert_eq!(
            iterator.size_hint(),
            (l.number_of_points(), Some(l.number_of_points()))
        );
        assert_eq!(iterator.size_hint(), (16, Some(16)));
        iterator.nth(9);
        assert_eq!(
            iterator.size_hint(),
            (
                l.number_of_points() - 10,       // 6
                Some(l.number_of_points() - 10)  // 6
            )
        );
        assert!(iterator.nth(4).is_some());
        assert_eq!(iterator.size_hint(), (1, Some(1)));
        assert!(iterator.next().is_some());
        assert_eq!(iterator.size_hint(), (0, Some(0)));
        assert!(iterator.next().is_none());
        assert_eq!(iterator.size_hint(), (0, Some(0)));

        Ok(())
    }
}
