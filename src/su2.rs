
//! module for SU(2) matrix

use super::{
    ZERO,
    ONE,
    I,
    CMatrix2,
    Real,
    lattice::Sign,
    Complex
};
use once_cell::sync::Lazy;
use rand_distr::Distribution;

/// First Pauli matrix.
///
/// ```math
/// 0   1
/// 1   0
/// ```
pub static PAULI_1: Lazy<CMatrix2> = Lazy::new(|| CMatrix2::new(
    ZERO, ONE,
    ONE, ZERO
));

/// Second Pauli matrix.
///
/// ```math
/// 0   -i
/// i    0
/// ```
pub static PAULI_2: Lazy<CMatrix2> = Lazy::new(|| CMatrix2::new(
    ZERO, -I,
    I, ZERO
));

/// Third Pauli matrix.
///
/// ```math
/// 1    0
/// 0   -1
/// ```
pub static PAULI_3: Lazy<CMatrix2> = Lazy::new(|| CMatrix2::new(
    ONE, ZERO,
    ZERO, -ONE
));

/// List of Pauli matrices, see [wikipedia](https://en.wikipedia.org/w/index.php?title=Pauli_matrices&oldid=1002053121)
pub static PAULI_MATRICES: Lazy<[&CMatrix2; 3]> = Lazy::new(||
    [&PAULI_1, &PAULI_2, &PAULI_3]
);

/// Get a radom SU(2) matrix close the origine.
///
/// Note that it diverge from SU(2) sligthly.
pub fn get_random_su2_close_to_unity(spread_parameter: Real, rng: &mut impl rand::Rng) -> CMatrix2 {
    let d = rand::distributions::Uniform::new(-1_f64, 1_f64);
    let r0 = d.sample(rng);
    let r = na::Vector3::<Real>::from_fn(|_, _| d.sample(rng));
    let x = r.try_normalize(f64::EPSILON).unwrap_or(r) * spread_parameter;
    let x0 = Sign::sign(r0).to_f64() * (1_f64 - spread_parameter.powi(2)).sqrt();
    CMatrix2::identity() * Complex::from(x0)
        + x.iter().enumerate().map(|(i, el)| PAULI_MATRICES[i] * Complex::from(el) ).sum::<CMatrix2>()
}
