
# v0.3.0

## breaking changes
- [`SymplecticEulerError`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/integrator/symplectic_euler/enum.SymplecticEulerError.html) is now marked `#[non_exhaustive]`
- change the feature "no-overflow-test" to "overflow-test" and enabling "overflow-test" by default

## non breaking
- update dependencies
- remove the extern crate declaration in lib.rs
- correct documentation

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