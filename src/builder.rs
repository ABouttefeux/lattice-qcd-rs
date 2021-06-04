//! builder utility

use std::num::NonZeroUsize;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Type of generation
#[non_exhaustive]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum GenType<'rng, Rng: rand::Rng + ?Sized> {
    /// Cold generation all ellements are set to the default
    Cold,
    /// Random deterministe
    #[cfg_attr(feature = "serde-serialize", serde(skip_deserializing))]
    HotDeterministe(&'rng mut Rng),
    /// Random deterministe but own the RNG (for instance the result of `clone`)
    HotDeterministeOwned(Box<Rng>),
    /// Random threaded (non deterministe)
    HotThreaded(NonZeroUsize),
}
