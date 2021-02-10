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
//! let mut mc = MCWrapper::new(MetropolisHastingsDeltaDiagnostic::new(number_of_rand, spread_parameter).unwrap(), rng);
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

extern crate nalgebra as na;
extern crate approx;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate crossbeam;
extern crate rayon;
#[cfg(feature = "serde-serialize")]
extern crate serde;

pub use na::ComplexField;
pub use rand_distr::Distribution;
pub use rand::{Rng, SeedableRng};

pub mod lattice;
pub mod su3;
pub mod su2;
pub mod field;
pub mod number;
pub mod integrator;
pub mod thread;
pub mod utils;
pub mod simulation;

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
