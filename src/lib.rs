
extern crate nalgebra as na;
extern crate approx;

pub mod lattice;
pub mod su3;
pub mod field;

#[cfg(test)]
mod test;

pub type Real = f64;
pub type Complex = na::Complex<f64>;

const ONE: Complex = Complex::new(1_f64, 0_f64);
const I: Complex = Complex::new(0_f64, 1_f64);
const ZERO: Complex = Complex::new(0_f64, 0_f64);
