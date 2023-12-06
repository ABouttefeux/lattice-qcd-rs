//! Module for [`IteratorLatticeLinkCanonical`]
//! # Example
// TODO

use super::{LatticeIterator, LatticeParIter};
use crate::lattice::LatticeLinkCanonical;

/// Iterator over [`LatticeLinkCanonical`] associated to a particular
/// [`crate::lattice::LatticeCyclic`].
/// # Example
pub type IteratorLatticeLinkCanonical<'a, const D: usize> =
    LatticeIterator<'a, D, LatticeLinkCanonical<D>>;

/// Parallel iterator over [`LatticeLinkCanonical`]
/// # Example
pub type ParIterLatticeLinkCanonical<'a, const D: usize> =
    LatticeParIter<'a, D, LatticeLinkCanonical<D>>;

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
