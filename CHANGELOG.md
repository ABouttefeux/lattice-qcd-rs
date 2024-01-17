
# v0.3.0

## dependencies

- updating to crossbeam 0.8.4
- updating to num-traits 0.2.17
- updating to rayon 1.8

## breaking changes
- update dependencies
- [`SymplecticEulerError`](hnew_deterministtps://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/integrator/symplectic_euler/enum.SymplecticEulerError.html) is now marked `#[non_exhaustive]`
- change the feature "no-overflow-test" to "overflow-test" and enabling "overflow-test" by default
- [`Su3Adjoint::random`] now takes the distribution as a generic parameter instead of an impl trait type and can be unsized
- [`EField::new_random`] now takes the distribution as a generic parameter instead of an impl trait type and can be unsized
- [`EField::new_determinist`] now takes the distribution as a generic parameter instead of an impl trait type and can be unsized
- [`LatticeStateDefault::new_determinist`] now takes the rng as a generic parameter instead of an impl trait type and can be unsized
- [`LatticeStateEFSyncDefault::new_random_e_state`] now takes the rng as a generic parameter instead of an impl trait type and can be unsized
- [`LatticeStateEFSyncDefault::new_determinist`] now takes the distribution as a generic parameter instead of an impl trait type and can be unsized
- [`IteratorLatticePoint::new`] and [`IteratorLatticeLinkCanonical::new`] are not constant anymore.
- rename [`IteratorDirection::new`] to [`IteratorDirection::new_with_first_element`] and [`IteratorLatticeLinkCanonical::new`] to [`IteratorLatticeLinkCanonical::new_with_first_element`]. Moreover introduce a function `new` that takes a single argument.
- rename [`DirectionEnum::to_index`] to [`DirectionEnum::index`] to avoid name colision with trait [`LatticeElementToIndex`]

## non breaking
- remove the extern crate declaration in lib.rs
- correct documentation
- add `#[inline]` to a all public facing function
- add methods [`inter`] and [`iter_mut`] to [`EField`]
- [`LatticeStateEFSyncDefault::new_random_threaded`] now takes unsized distribution.
- [`LinkMatrix`::iter`] and [`LinkMatrix::iter_mut`] now also implements [`DoubleEndedIterator`]
- moved the definition of [`DirectionConversionError`] out of the proc macro into lattice.rs
- add comments on some private items
- organize internal code in private submodule with reexport
- depreciate [`random_matrix_2`]
- depreciate [`FixedPointNumber`] and [`I32U128`] which never implemented
- implemented the trait [`DoubleEndedIterator`]for [`IteratorDirection`], [`IteratorLatticeLinkCanonical`] and [`IteratorLatticePoint`],
- implemented [`LatticeElementToIndex`] for [`DirectionEnum`].
- introduced a private Sealed trait.
- introduced sealed trait [`RandomAccessIterator`], [`NumberOfLatticeElement`] and [`IndexToElement`].
- introduce structure [`LatticeIterator`].
- [`IteratorLatticeLinkCanonical`] and [`IteratorLatticePoint`] are now aliases for the new structure [`LatticeIterator`] (they should work basically the same way as before).
- implemented , [`rayon::iter::ParallelIterator`], [`rayon::iter::IndexedParallelIterator`] for [`LatticeParIter`] and [`IteratorDirection`].
- [`LatticeIterator`] and [`LatticeParIter`] and be easily converted into each other.
- implemented [`LatticeElementToIndex`] for [`DirectionEnum`].
- added [`LatticeCyclic::par_iter_links`] and [`LatticeCyclic::par_iter_points`] to get parallel iterator on the links and points respectively.

# v0.2.1

- add const on multiple functions where applicable
- update nalgebra to 0.31.0
- update num-traits to 0.2.15
- update rayon to 1.5.3
 

# v0.2.0

First release

- thread::ThreadError::Panic now contain a vector of error
- fixe issue with the threading module's function giving inconsistent error type
- iterator opaque type returned now also implement ExactSizeIterator and FusedIterator
- add licenses MIT or APACHE-2.0
- add documentation
- correct doctest Su3Adjoint::as_mut_vector
- migrate to rust 2021
- use readme in libraries root documentation
- rename lots of getter to match C-GETTER convention
- rename some structure for C-WORD-ORDER
- Rename `LatticeCylcique` to `LatticeCyclic`
- Rename `SimulationStateSynchrone` to `SimulationStateSynchronous`
- Rename structures and functions to correct spelling mistakes