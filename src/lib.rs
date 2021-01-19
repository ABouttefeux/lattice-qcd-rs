
extern crate nalgebra as na;
extern crate approx;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
//extern crate t1ha;

pub mod lattice;
pub mod su3;
pub mod field;
pub mod number;
pub mod integrator;

#[cfg(test)]
mod test;

pub type Real = f64;
pub type Complex = na::Complex<Real>;
pub type Vector8<N> = na::VectorN<N, na::U8>;
pub type CMatrix3 = na::Matrix3<Complex>;


const ONE: Complex = Complex::new(1_f64, 0_f64);
const I: Complex = Complex::new(0_f64, 1_f64);
const ZERO: Complex = Complex::new(0_f64, 0_f64);
