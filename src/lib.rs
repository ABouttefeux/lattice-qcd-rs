//! ![Lattice QCD rs](https://raw.githubusercontent.com/ABouttefeux/lattice-qcd-rs/develop/logo.svg)
//!
//! Classical lattice QCD simulation and tools.
//!
//! ## Usage
//!
//! Add `lattice_qcd_rs = { version = "0.1.0", git = "https://github.com/ABouttefeux/lattice_qcd_rs", branch = "develop" }`
//! into your `cargo.toml`.
//!
//! ## Why
//! This some code for my PhD thesis.
//! Mainly I use [arXiv:0707.2458](https://arxiv.org/abs/0707.2458),
//! [arXiv:0902.28568](https://arxiv.org/abs/0707.2458) and
//! [arXiv:2010.07316](https://arxiv.org/abs/2010.07316) as a basis.
//!
//! ## Goal
//! The goal is to provide an easy to use, fast and safe library to do classical lattice simulation.
//!
//! ## Discussion about Random Number Generators (RNGs)
//!
//! This library use the trait [`rand::RngCore`] any time a random number generator.
//! The choice of RNG is up to the user of the library. However there is a few trade offs to consider.
//!
//! Let us break the different generator into cathegories.
//! For more details see (https://rust-random.github.io/book/guide-gen.html)
//!
//! Some of the possible choice :
//! - **Recomanded** [`rand_xoshiro::Xoshiro256PlusPlus`](https://docs.rs/rand_xoshiro/0.6.0/rand_xoshiro/struct.Xoshiro256PlusPlus.html)
//! Non-cryptographic. It has good performance and stastistical quality, reproducible, and has useful `jump` function.
//! It is the recomanded PRNG.
//! - [`rand::rngs::ThreadRng`] a CSPRNG. The data is not repoducible and it is resseded often. It is however slow.
//! - [`rand::rngs::StdRng`] cryptographic secure, can be seeded.
//! It is deterministique but not repoducible between platform. It is however slow.
//! - [`rand_jitter::JitterRng`](https://docs.rs/rand_jitter/0.3.0/rand_jitter/) True RNG but very slow.
//!
//! # Examples
//! ```
//! use lattice_qcd_rs::{
//!    simulation::state::{LatticeStateDefault, LatticeState},
//!    simulation::monte_carlo::{MCWrapper, MetropolisHastingsDeltaDiagnostic},
//!    ComplexField,
//! };
//!
//! let mut rng = rand::thread_rng();
//!
//! let size = 1_000_f64;
//! let number_of_pts = 4;
//! let beta = 2_f64;
//! let mut simulation = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();
//!
//! let number_of_rand = 20;
//! let spread_parameter = 1E-5_f64;
//! let mut mc = MetropolisHastingsDeltaDiagnostic::new(number_of_rand, spread_parameter, rng).unwrap();
//!
//! let number_of_sims = 100;
//! for _ in 0..number_of_sims / 10 {
//!     for _ in 0..10 {
//!         simulation = simulation.monte_carlo_step(&mut mc).unwrap();
//!     }
//!     simulation.normalize_link_matrices(); // we renormalize all matrices back to SU(3);
//! };
//! let average = simulation.average_trace_plaquette().unwrap().real();
//! ```
//! Alternatively other Monte Carlo algorithm can be used like,
//! ```
//! use lattice_qcd_rs::{
//!    simulation::state::{LatticeStateDefault, LatticeState},
//!    simulation::monte_carlo::{MCWrapper, MetropolisHastingsDiagnostic},
//! };
//!
//! let mut rng = rand::thread_rng();
//!
//! let size = 1_000_f64;
//! let number_of_pts = 4;
//! let beta = 2_f64;
//! let mut simulation = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();
//!
//! let number_of_rand = 20;
//! let spread_parameter = 1E-5_f64;
//! let mut mc = MCWrapper::new(
//!     MetropolisHastingsDiagnostic::new(
//!         number_of_rand,
//!         spread_parameter
//!     ).unwrap(),
//!     rng
//! );
//!
//! simulation = simulation.monte_carlo_step(&mut mc).unwrap();
//! simulation.normalize_link_matrices();
//! ```
//! or
//! ```
//! use lattice_qcd_rs::{
//!    simulation::state::{LatticeStateDefault, LatticeState},
//!    simulation::monte_carlo::HybridMonteCarloDiagnostic,
//!    integrator::SymplecticEulerRayon,
//! };
//!
//! let mut rng = rand::thread_rng();
//!
//! let size = 1_000_f64;
//! let number_of_pts = 4;
//! let beta = 2_f64;
//! let mut simulation = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();
//!
//! let delta_t = 1E-3_f64;
//! let number_of_step = 10;
//! let mut mc = HybridMonteCarloDiagnostic::new(
//!     delta_t,
//!     number_of_step,
//!     SymplecticEulerRayon::new(),
//!     rng
//! );
//!
//! simulation = simulation.monte_carlo_step(&mut mc).unwrap();
//! simulation.normalize_link_matrices();
//! ```

#![allow(clippy::needless_return)]
#![warn(clippy::as_conversions)]
#![warn(clippy::cast_lossless)]
//#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
//#![warn(clippy::default_numeric_fallback)]
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
//#![warn(clippy::semicolon_if_nothing_returned)]
#![warn(clippy::suboptimal_flops)]
#![warn(clippy::todo)]
#![warn(clippy::trivially_copy_pass_by_ref)]
//#![warn(clippy::type_repetition_in_bounds)]
#![warn(clippy::unreadable_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::unused_self)]

//#![warn(clippy::missing_errors_doc)]
//#![warn(missing_docs)]


extern crate nalgebra as na;
extern crate approx;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate crossbeam;
extern crate rayon;
#[cfg(feature = "serde-serialize")]
extern crate serde;
extern crate lattice_qcd_rs_procedural_macro;

pub use na::ComplexField;
pub use rand_distr::Distribution;
pub use rand::{Rng, SeedableRng};

#[macro_use]
mod macro_def;
pub mod lattice;
pub mod su3;
pub mod su2;
pub mod field;
pub mod number;
pub mod integrator;
pub mod thread;
pub mod utils;
pub mod simulation;
pub mod dim;

#[cfg(test)]
mod test;

/// alias for [`f64`]
pub type Real = f64;
/// easy to use allias for [`nalgebra::Complex::<Real>`]
pub type Complex = na::Complex<Real>;
/// alias for [`nalgebra::VectorN::<N, nalgebra::U8>`]
pub type Vector8<N> = na::VectorN<N, na::U8>;
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
