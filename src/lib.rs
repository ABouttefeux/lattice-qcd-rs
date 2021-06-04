#![doc = include_str!("../README.md")]
//
//#![warn(clippy::as_conversions)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::implicit_hasher)]
#![warn(clippy::implicit_saturating_sub)]
#![warn(clippy::imprecise_flops)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::macro_use_imports)]
#![warn(clippy::manual_ok_or)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::non_ascii_literal)]
#![warn(clippy::semicolon_if_nothing_returned)]
#![warn(clippy::suboptimal_flops)]
#![warn(clippy::todo)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::type_repetition_in_bounds)]
#![warn(clippy::unreadable_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::unused_self)]
#![warn(clippy::unnecessary_wraps)]
#![warn(clippy::missing_errors_doc)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![doc(html_root_url = "https://docs.rs/lattice_qcd_rs/0.2.0")]

extern crate approx;
extern crate crossbeam;
extern crate lattice_qcd_rs_procedural_macro;
extern crate nalgebra as na;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate rayon;
#[cfg(feature = "serde-serialize")]
extern crate serde;

pub use na::ComplexField;
pub use rand::{Rng, SeedableRng};
pub use rand_distr::Distribution;

#[macro_use]
mod macro_def;
pub(crate) mod builder;
pub mod dim;
pub mod error;
pub mod field;
pub mod integrator;
pub mod lattice;
pub mod number;
pub mod prelude;
pub mod simulation;
pub mod statistics;
pub mod su2;
pub mod su3;
pub mod thread;
pub mod utils;

#[cfg(test)]
mod test;

/// alias for [`f64`]
pub type Real = f64;
/// easy to use alias for [`nalgebra::Complex::<Real>`]
pub type Complex = na::Complex<Real>;
/// alias for [`nalgebra::SVector::<N, nalgebra::U8>`]
pub type Vector8<N> = na::SVector<N, 8>;
/// alias for [`nalgebra::Matrix3<nalgebra::Complex>`]
pub type CMatrix3 = na::Matrix3<Complex>;
/// alias for [`nalgebra::Matrix2<nalgebra::Complex>`]
pub type CMatrix2 = na::Matrix2<Complex>;

/// Complex 1
const ONE: Complex = Complex::new(1_f64, 0_f64);
/// Complex I
const I: Complex = Complex::new(0_f64, 1_f64);
/// Complex 0
const ZERO: Complex = Complex::new(0_f64, 0_f64);
