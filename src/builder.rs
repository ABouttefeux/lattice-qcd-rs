//! builder utility

use std::num::NonZeroUsize;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Type of generation
#[non_exhaustive]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum GenType<'rng, Rng: rand::Rng + ?Sized> {
    /// Cold generation all elements are set to the default
    Cold,
    /// Random determinist
    #[cfg_attr(feature = "serde-serialize", serde(skip_deserializing))]
    HotDeterminist(&'rng mut Rng),
    /// Random determinist but own the RNG (for instance the result of `clone`)
    HotDeterministOwned(Box<Rng>),
    /// Random threaded (non determinist)
    HotThreaded(NonZeroUsize),
}
