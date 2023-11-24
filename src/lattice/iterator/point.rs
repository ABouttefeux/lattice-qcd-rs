//! Module for [`IteratorLatticePoint`]

use super::{super::LatticePoint, LatticeIterator};

/// Iterator over [`LatticePoint`]
pub type IteratorLatticePoint<'a, const D: usize> = LatticeIterator<'a, D, LatticePoint<D>>;

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
