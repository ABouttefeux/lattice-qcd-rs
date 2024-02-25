//! Utils functions and structures.
//!
//! Mainly things that I do not know where to put.

mod factorial;
mod sign;

#[doc(inline)]
pub(crate) use factorial::FactorialNumber;
pub use factorial::{factorial, FactorialStorageDyn, MAX_NUMBER_FACTORIAL};
pub use sign::{levi_civita, Sign};
