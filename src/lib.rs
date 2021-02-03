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
//! # Example
//! let us create a basic random state and let us simulate.
//! ```
//! extern crate rand;
//! extern crate rand_distr;
//! use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationStateSync;
//! use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationState;
//! use lattice_qcd_rs::integrator::SymplecticEuler;
//!
//! let mut rng = rand::thread_rng();
//! let distribution = rand::distributions::Uniform::from(
//!     -std::f64::consts::PI..std::f64::consts::PI
//! );
//! let state1 = LatticeHamiltonianSimulationStateSync::new_deterministe(100_f64, 1_f64, 8, &mut rng, &distribution)
//!     .unwrap();
//! let state2 = state1.simulate(0.0001_f64, &SymplecticEuler::new(8)).unwrap();
//! let state3 = state2.simulate(0.0001_f64, &SymplecticEuler::new(8)).unwrap();
//! ```
//! Let us then compute and compare the Hamiltonian.
//! ```
//! # extern crate rand;
//! # extern crate rand_distr;
//! # use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationStateSync;
//! # use lattice_qcd_rs::simulation::LatticeHamiltonianSimulationState;
//! # use lattice_qcd_rs::integrator::SymplecticEuler;
//! #
//! # let mut rng = rand::thread_rng();
//! # let distribution = rand::distributions::Uniform::from(
//! #    -std::f64::consts::PI..std::f64::consts::PI
//! # );
//! # let state1 = LatticeHamiltonianSimulationStateSync::new_deterministe(100_f64, 1_f64, 8, &mut rng, &distribution)
//! #     .unwrap();
//! # let state2 = state1.simulate(0.0001_f64, &SymplecticEuler::new(8)).unwrap();
//! # let state3 = state2.simulate(0.0001_f64, &SymplecticEuler::new(8)).unwrap();
//! let h = state1.get_hamiltonian_total();
//! let h2 = state3.get_hamiltonian_total();
//! println!("The error on the Hamiltonian is {}", h - h2);
//! ```

#![allow(clippy::needless_return)]

extern crate nalgebra as na;
extern crate approx;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate crossbeam;
extern crate rayon;

pub mod lattice;
pub mod su3;
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

/// Complex 1
const ONE: Complex = Complex::new(1_f64, 0_f64);
/// Complex I
const I: Complex = Complex::new(0_f64, 1_f64);
/// Complex 0
const ZERO: Complex = Complex::new(0_f64, 0_f64);

// TODO refactor pos on lattice without time
