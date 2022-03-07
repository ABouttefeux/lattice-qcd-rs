
# v0.2.0

First release

- thread::ThreadError::Panic now contain a vector of error
- fixe issue with the threading module's function giving inconsistent error type
- iterator opac type returned now also implement ExactSizeIterator and FusedIterator
- add licences MIT or APACHE-2.0
- add documentation
- correct doctest Su3Adjoint::as_mut_vector
- migrate to rust 2021
- use readme in librairies root documentation
- rename lots of gettet to match C-GETTER convention