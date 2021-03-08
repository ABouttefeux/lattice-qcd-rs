
//! module for SU(2) matrix

use super::{
    ZERO,
    ONE,
    I,
    CMatrix2,
    Real,
    Complex
};
use once_cell::sync::Lazy;
use rand_distr::Distribution;

/// First Pauli matrix.
///
/// ```textrust
/// 0   1
/// 1   0
/// ```
pub static PAULI_1: Lazy<CMatrix2> = Lazy::new(|| CMatrix2::new(
    ZERO, ONE,
    ONE, ZERO
));

/// Second Pauli matrix.
///
/// ```textrust
/// 0  -i
/// i   0
/// ```
pub static PAULI_2: Lazy<CMatrix2> = Lazy::new(|| CMatrix2::new(
    ZERO, -I,
    I, ZERO
));

/// Third Pauli matrix.
///
/// ```textrust
/// 1   0
/// 0  -1
/// ```
pub static PAULI_3: Lazy<CMatrix2> = Lazy::new(|| CMatrix2::new(
    ONE, ZERO,
    ZERO, -ONE
));

/// List of Pauli matrices, see
/// [wikipedia](https://en.wikipedia.org/w/index.php?title=Pauli_matrices&oldid=1002053121)
pub static PAULI_MATRICES: Lazy<[&CMatrix2; 3]> = Lazy::new(||
    [&PAULI_1, &PAULI_2, &PAULI_3]
);

/// Get a radom SU(2) matrix close the 1 or -1.
///
/// Note that it diverges from SU(2) sligthly.
/// `spread_parameter` should be between between 0 and 1 both excluded to generate valide data.
/// outside this bound it will not panic but can have unexpected results.
pub fn get_random_su2_close_to_unity<R>(spread_parameter: Real, rng: &mut R) -> CMatrix2
    where R: rand::Rng + ?Sized,
{
    let d = rand::distributions::Uniform::new(-1_f64, 1_f64);
    let r = na::Vector3::<Real>::from_fn(|_, _| d.sample(rng));
    let x = r.try_normalize(f64::EPSILON).unwrap_or(r) * spread_parameter;
    let d_sign = rand::distributions::Bernoulli::new(0.5_f64).unwrap();
    // we could have use the spread_parameter but it is safer to use the norm of x
    let x0_unsigned = (1_f64 - x.norm_squared()).sqrt();
    let x0;
    if d_sign.sample(rng) {
        x0 = x0_unsigned;
    }
    else{
        x0 = - x0_unsigned;
    }
    
    get_complex_matrix_from_vec(x0, x)
}

/// return x0 1 + i x_i * \sigma_i
pub fn get_complex_matrix_from_vec(x0: Real, x: na::Vector3<Real>) -> CMatrix2 {
    CMatrix2::identity() * Complex::from(x0) +
        x.iter().enumerate().map(|(i, el)| PAULI_MATRICES[i] * Complex::new(0_f64, *el)).sum::<CMatrix2>()
}
