//! module for SU(2) matrix

use rand_distr::Distribution;

use super::{CMatrix2, Complex, ComplexField, Real, I, ONE, ZERO};

/// First Pauli matrix.
///
/// ```textrust
/// 0   1
/// 1   0
/// ```
pub const PAULI_1: CMatrix2 = CMatrix2::new(ZERO, ONE, ONE, ZERO);

/// Second Pauli matrix.
///
/// ```textrust
/// 0  -i
/// i   0
/// ```
pub const PAULI_2: CMatrix2 = CMatrix2::new(ZERO, Complex::new(0_f64, -1_f64), I, ZERO);

/// Third Pauli matrix.
///
/// ```textrust
/// 1   0
/// 0  -1
/// ```
pub const PAULI_3: CMatrix2 = CMatrix2::new(ONE, ZERO, ZERO, Complex::new(1_f64, 0_f64));

/// List of Pauli matrices, see
/// [wikipedia](https://en.wikipedia.org/w/index.php?title=Pauli_matrices&oldid=1002053121)
pub const PAULI_MATRICES: [&CMatrix2; 3] = [&PAULI_1, &PAULI_2, &PAULI_3];

/// Get a radom SU(2) matrix close the 1 or -1.
///
/// Note that it diverges from SU(2) sligthly.
/// `spread_parameter` should be between between 0 and 1 both excluded to generate valide data.
/// outside this bound it will not panic but can have unexpected results.
pub fn get_random_su2_close_to_unity<R>(spread_parameter: Real, rng: &mut R) -> CMatrix2
where
    R: rand::Rng + ?Sized,
{
    let d = rand::distributions::Uniform::new(-1_f64, 1_f64);
    let r = na::Vector3::<Real>::from_fn(|_, _| d.sample(rng));
    let x = r.try_normalize(f64::EPSILON).unwrap_or(r) * spread_parameter;
    let d_sign = rand::distributions::Bernoulli::new(0.5_f64).unwrap();
    // we could have use the spread_parameter but it is safer to use the norm of x
    let x0_unsigned = (1_f64 - x.norm_squared()).sqrt();
    // determine the sign of x0.
    let x0 = if d_sign.sample(rng) {
        x0_unsigned
    }
    else {
        -x0_unsigned
    };

    get_complex_matrix_from_vec(x0, x)
}

/// Return `x0 1 + i x_i * \sigma_i`.
pub fn get_complex_matrix_from_vec(x0: Real, x: na::Vector3<Real>) -> CMatrix2 {
    CMatrix2::identity() * Complex::from(x0)
        + x.iter()
            .enumerate()
            .map(|(i, el)| PAULI_MATRICES[i] * Complex::new(0_f64, *el))
            .sum::<CMatrix2>()
}

/// Take any 2x2 matrix and project it to a matric `X` such that `X / X.determinant().modulus().sqrt()`
/// is SU(2).
pub fn project_to_su2_unorm(m: CMatrix2) -> CMatrix2 {
    m - m.adjoint() + CMatrix2::identity() * m.trace().conjugate()
}

/// Project the matrix to SU(2). Return the identity if the norm after unormalize is
/// subnormal (see[`f64::is_normal`]).
pub fn project_to_su2(m: CMatrix2) -> CMatrix2 {
    let m = project_to_su2_unorm(m);
    if m.determinant().modulus().is_normal() {
        m / Complex::from(m.determinant().modulus().sqrt())
    }
    else {
        CMatrix2::identity()
    }
}

/// Get an Uniformly random SU(2) matrix.
pub fn get_random_su2(rng: &mut impl rand::Rng) -> CMatrix2 {
    let d = rand::distributions::Uniform::new(-1_f64, 1_f64);
    let mut random_vector = na::Vector2::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)));
    while !random_vector.norm().is_normal() {
        random_vector = na::Vector2::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)))
    }
    let vector_normalize = random_vector / Complex::from(random_vector.norm());
    CMatrix2::new(
        vector_normalize[0],
        vector_normalize[1],
        -vector_normalize[1].conjugate(),
        vector_normalize[0].conjugate(),
    )
}

/// Return wether the input matrix is SU(2) up to epsilon.
pub fn is_matrix_su2(m: &CMatrix2, epsilon: f64) -> bool {
    ((m.determinant() - Complex::from(1_f64)).modulus_squared() < epsilon)
        && ((m * m.adjoint() - CMatrix2::identity()).norm() < epsilon)
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    use super::*;

    const EPSILON: f64 = 0.000_000_001_f64;
    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    fn test_u2_const() {
        // test constant
        for el in &PAULI_MATRICES {
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

            assert_matrix_is_su_2!((p / p.determinant().sqrt()), EPSILON);
        }

        for _ in 0..100 {
            let r = CMatrix2::from_fn(|_, _| Complex::new(d.sample(&mut rng), d.sample(&mut rng)));
            let p = project_to_su2(r);
            assert_matrix_is_su_2!(p, EPSILON);
        }
    }

    #[test]
    fn random_su2() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        for _ in 0..100 {
            let m = get_random_su2(&mut rng);
            assert_matrix_is_su_2!(m, EPSILON);
        }
    }
}
