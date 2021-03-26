
//! module for SU(2) matrix

use super::{
    ZERO,
    ONE,
    I,
    CMatrix2,
    Real,
    Complex,
    ComplexField,
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

/// Take any 2x2 matrix and project it to a matric `X` such that `X / X.determinant().sqrt()` is SU(2).
pub fn project_to_su2_unorm(m: CMatrix2) -> CMatrix2 {
    m - m.adjoint() + CMatrix2::identity() * m.trace().conjugate()
}

/// return wether the input matrix is SU(2) up to epsilon.
pub fn is_matrix_su2(m: &CMatrix2, epsilon: f64) -> bool {
    ((m.determinant() - Complex::from(1_f64)).modulus_squared() < epsilon) &&
    ((m * m.adjoint() - CMatrix2::identity()).norm() < epsilon)
}

#[cfg(test)]
mod test {
    use super::*;
    use rand_distr::Distribution;
    use rand::SeedableRng;
    
    const EPSILON: f64 = 0.000_000_001_f64;
    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;
    
    #[test]
    fn test_su2_const() {
        // test constant
        for el in &*PAULI_MATRICES {
            assert_matrix_is_unitary_2!(*el, EPSILON);
        }
    }
        
    #[test]
    fn test_su2_project() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let d = rand::distributions::Uniform::new(-1_f64, 1_f64);
        for _ in 0..100 {
            let r = CMatrix2::from_fn(|_, _| Complex::new(d.sample(&mut rng), d.sample(&mut rng)));
            let p = project_to_su2_unorm(r);
            assert!(p.trace().imaginary().abs() < EPSILON);
            assert!((p * p.adjoint() - CMatrix2::identity() * p.determinant()).norm() < EPSILON);
            
            assert_matrix_is_su_2!((p/ p.determinant().sqrt()), EPSILON);
        }
        
    }
}
