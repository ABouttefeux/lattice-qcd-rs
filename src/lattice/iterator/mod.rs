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

//---------------------------------------
// mod

mod direction;
mod double_ended_counter;
mod element;
mod lattice_iterator;
mod link_canonical;
mod parallel_iterator;
mod point;
mod producer;

//---------------------------------------
// uses

pub use self::direction::IteratorDirection;
pub use self::double_ended_counter::DoubleEndedCounter;
pub use self::element::IteratorElement;
pub use self::lattice_iterator::LatticeIterator;
pub use self::link_canonical::{IteratorLatticeLinkCanonical, ParIterLatticeLinkCanonical};
pub use self::parallel_iterator::ParIter;
pub use self::point::{IteratorLatticePoint, ParIterLatticePoint};
use crate::private::Sealed;

//---------------------------------------
// Trait RandomAccessIterator definition

/// Trait for generic implementation of [`Iterator`] for implementor of this trait.
///
/// It has a notion of dimension as it is link to the notion of lattice element.
/// And a lattice takes a dimension.
///
/// This trait is a super trait of [`Sealed`] which is private meaning that It can't be
/// implemented outside of this trait.
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

/// Trait used by [`producer::LatticeProducer`] for implementing
/// [`rayon::iter::plumbing::Producer`].
/// It is separate trait than [`RandomAccessIterator`] to avoid more a [`Clone`] constrain.
///
/// [`Split::split_at`] return two self and not two [`producer::LatticeProducer`] the idea is that they
/// should be converted into [`producer::LatticeProducer`].
///
/// This trait is a super trait of [`Sealed`] which is private meaning that It can't be
/// implemented outside of this trait.
trait Split: RandomAccessIterator + Sized + Sealed {
    /// See [`rayon::iter::plumbing::Producer::split_at`]
    #[must_use]
    fn split_at(self, index: usize) -> (Self, Self);
}

// TODO remove
// impl<T, I> Split for T
// where
//     T: RandomAccessIterator<Item = I> + Clone + AsMut<DoubleEndedCounter<I>>,
//     I: Clone,
// {
//     fn split_at(self, index: usize) -> (Self, Self) {
//         let splinting = self.increase_front_element_by(index);
//         let mut first = self.clone();
//         *first.as_mut().end_mut() = splinting.clone();
//         let mut second = self;
//         *second.as_mut().front_mut() = splinting;
//         (first, second)
//     }
// }

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
