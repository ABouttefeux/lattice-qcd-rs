//! Module for SU(2) matrices.

use nalgebra::Vector3;
use rand::Rng;
use rand_distr::{Bernoulli, Distribution, Uniform};

use super::{CMatrix2, Complex, ComplexField, Real, I, ONE, ZERO};

/// First Pauli matrix. See [`PAULI_MATRICES`].
///
/// Its value is
/// ```textrust
/// 0   1
/// 1   0
/// ```
pub const PAULI_1: CMatrix2 = CMatrix2::new(
    ZERO, ONE, // ---
    ONE, ZERO,
);

/// Second Pauli matrix. See [`PAULI_MATRICES`].
///
/// Its value is
/// ```textrust
/// 0  -i
/// i   0
/// ```
pub const PAULI_2: CMatrix2 = CMatrix2::new(
    ZERO,
    Complex::new(0_f64, -1_f64),
    // ---
    I,
    ZERO,
);

/// Third Pauli matrix. See [`PAULI_MATRICES`].
///
/// Its value is
/// ```textrust
/// 1   0
/// 0  -1
/// ```
pub const PAULI_3: CMatrix2 = CMatrix2::new(
    ONE,
    ZERO,
    // ---
    ZERO,
    Complex::new(1_f64, 0_f64),
);

/// List of Pauli matrices, see
/// [wikipedia](https://en.wikipedia.org/w/index.php?title=Pauli_matrices&oldid=1002053121).
pub const PAULI_MATRICES: [&CMatrix2; 3] = [&PAULI_1, &PAULI_2, &PAULI_3];

/// Get a radom SU(2) matrix close the 1 or -1.
///
/// Note that it diverges from SU(2) slightly.
/// `spread_parameter` should be between between 0 and 1 both excluded to generate valid data.
/// outside this bound it will not panic but can have unexpected results.
///
/// # Example
/// ```
/// # use lattice_qcd_rs::{assert_matrix_is_su_2,su2::random_su2_close_to_unity};
/// # use rand::SeedableRng;
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// for _ in 0..10 {
///     assert_matrix_is_su_2!(
///         random_su2_close_to_unity(0.000_000_001_f64, &mut rng),
///         0.000_000_1_f64
///     );
/// }
/// ```
/// but it will be not close to SU(2) up to [`f64::EPSILON`].
/// ```should_panic
/// # use lattice_qcd_rs::{assert_matrix_is_su_2,su2::random_su2_close_to_unity};
/// # use rand::SeedableRng;
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// assert_matrix_is_su_2!(
///     random_su2_close_to_unity(0.000_000_001_f64, &mut rng),
///     f64::EPSILON * 40_f64
/// );
/// ```
#[allow(clippy::missing_panics_doc)] // the expect nevers panic
#[inline]
#[must_use]
pub fn random_su2_close_to_unity<R>(spread_parameter: Real, rng: &mut R) -> CMatrix2
where
    R: rand::Rng + ?Sized,
{
    let distribution: Uniform<f64> = Uniform::new(-1_f64, 1_f64);
    let random_vector = Vector3::<Real>::from_fn(|_, _| distribution.sample(rng));
    let x = random_vector
        .try_normalize(f64::EPSILON)
        .unwrap_or(random_vector)
        * spread_parameter;
    // always exists, unwrap is safe
    let d_sign = Bernoulli::new(0.5_f64).expect("always exist");
    // we could have use the spread_parameter but it is safer to use the norm of x
    let x0_unsigned = (1_f64 - x.norm_squared()).sqrt();
    // determine the sign of x0.
    let x0 = if d_sign.sample(rng) {
        x0_unsigned
    } else {
        -x0_unsigned
    };

    complex_matrix_from_vec(x0, x)
}

/// Return `x0 1 + i x_i * \sigma_i`.
///
/// # Examples
/// ```
/// use lattice_qcd_rs::{
///     assert_eq_matrix,
///     su2::{complex_matrix_from_vec, PAULI_1},
///     CMatrix2,
/// };
/// use nalgebra::Vector3;
///
/// let m = complex_matrix_from_vec(1.0, Vector3::new(0_f64, 0_f64, 0_f64));
/// assert_eq_matrix!(
///     m,
///     CMatrix2::new(
///         nalgebra::Complex::new(1_f64, 0_f64),
///         nalgebra::Complex::new(0_f64, 0_f64),
///         nalgebra::Complex::new(0_f64, 0_f64),
///         nalgebra::Complex::new(1_f64, 0_f64)
///     ),
///     f64::EPSILON
/// );
///
/// let m = complex_matrix_from_vec(0.5_f64, Vector3::new(1_f64, 0_f64, 0_f64));
/// let m2 = CMatrix2::new(
///     nalgebra::Complex::new(1_f64, 0_f64),
///     nalgebra::Complex::new(0_f64, 0_f64),
///     nalgebra::Complex::new(0_f64, 0_f64),
///     nalgebra::Complex::new(1_f64, 0_f64),
/// ) * nalgebra::Complex::new(0.5_f64, 0_f64)
///     + PAULI_1 * nalgebra::Complex::new(0_f64, 1_f64);
/// assert_eq_matrix!(m, m2, f64::EPSILON);
/// ```
#[inline]
#[must_use]
pub fn complex_matrix_from_vec(x0: Real, x: Vector3<Real>) -> CMatrix2 {
    CMatrix2::identity() * Complex::from(x0)
        + x.into_iter() // no way to move out the content without conversion it ? into_iter is the same as iter (or iter_mut)
            .enumerate()
            .map(|(i, el)| PAULI_MATRICES[i] * Complex::new(0_f64, *el))
            .sum::<CMatrix2>()
}

/// Take any 2x2 matrix and project it to a matrix `X` such that `X / X.determinant().modulus().sqrt()`
/// is SU(2) provided that the matrix's determinant is not zero.
///
/// # Examples
/// see [`project_to_su2`]
/// ```
/// # use lattice_qcd_rs::{su2::{project_to_su2_unorm, random_su2},CMatrix2,  Complex, assert_eq_matrix};
/// # use rand::SeedableRng;
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// let m = CMatrix2::zeros();
/// assert_eq_matrix!(project_to_su2_unorm(m), m, f64::EPSILON);
/// ```
// TODO more example
#[inline]
#[must_use]
pub fn project_to_su2_unorm(m: CMatrix2) -> CMatrix2 {
    m - m.adjoint() + CMatrix2::identity() * m.trace().conjugate()
}

/// Project the matrix to SU(2). Return the identity if the norm after unormalize is
/// subnormal (see[`f64::is_normal`]).
///
/// # Examples
/// ```
/// # use lattice_qcd_rs::{su2::{project_to_su2, random_su2, random_matrix_2},CMatrix2,  Complex, assert_eq_matrix, assert_matrix_is_su_2};
/// # use rand::SeedableRng;
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// let m = CMatrix2::zeros();
/// assert_eq_matrix!(project_to_su2(m), CMatrix2::identity(), f64::EPSILON);
/// for _ in 0..10 {
///     let m = random_su2(&mut rng);
///     assert_eq_matrix!(project_to_su2(m * Complex::new(0.5_f64, 0_f64)), m, 4_f64 * f64::EPSILON);
///     assert_eq_matrix!(project_to_su2(m), m, 4_f64 * f64::EPSILON);
///     assert_matrix_is_su_2!(project_to_su2(m), 4_f64 * f64::EPSILON);
/// }
/// for _ in 0..10 {
///     let m = random_matrix_2(&mut rng);
///     assert_matrix_is_su_2!(project_to_su2(m), 4_f64 * f64::EPSILON)
/// }
/// ```
#[inline]
#[must_use]
pub fn project_to_su2(m: CMatrix2) -> CMatrix2 {
    let m = project_to_su2_unorm(m);
    if m.determinant().modulus().is_normal() {
        m / Complex::from(m.determinant().modulus().sqrt())
    } else {
        CMatrix2::identity()
    }
}

/// Get an Uniformly random SU(2) matrix.
///
/// # Example
/// ```
/// # use lattice_qcd_rs::{assert_matrix_is_su_2,su2::random_su2};
/// # use rand::SeedableRng;
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// for _ in 0..10 {
///     assert_matrix_is_su_2!(random_su2(&mut rng), 4_f64 * f64::EPSILON);
/// }
#[inline]
#[must_use]
pub fn random_su2<Rng>(rng: &mut Rng) -> CMatrix2
where
    Rng: rand::Rng + ?Sized,
{
    let d = Uniform::new(-1_f64, 1_f64);
    let mut random_vector =
        nalgebra::Vector2::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)));
    while !random_vector.norm().is_normal() {
        random_vector =
            nalgebra::Vector2::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)));
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
///
/// # Example
/// ```
/// # use lattice_qcd_rs::{su2::{is_matrix_su2, random_su2}, CMatrix2};
/// # use rand::SeedableRng;
/// # use nalgebra::{Complex};
/// # let mut rng = rand::rngs::StdRng::seed_from_u64(0);
///
/// assert!(is_matrix_su2(&random_su2(&mut rng), 4_f64 * f64::EPSILON));
/// assert!(!is_matrix_su2(&CMatrix2::zeros(), 4_f64 * f64::EPSILON));
/// assert!(!is_matrix_su2(
///     &(random_su2(&mut rng) * Complex::new(0.5_f64, 1.7_f64)),
///     4_f64 * f64::EPSILON
/// ));
/// ```
#[inline]
#[must_use]
pub fn is_matrix_su2(m: &CMatrix2, epsilon: f64) -> bool {
    ((m.determinant() - Complex::from(1_f64)).modulus_squared() < epsilon)
        && ((m * m.adjoint() - CMatrix2::identity()).norm() < epsilon)
}

#[deprecated(
    since = "0.3.0",
    note = "the uniform distribution does not represent any meaningful representation of the matrices"
)]
#[doc(hidden)] // it does not really have a use
/// Crate a random 2x2 Matrix with a uniform distribution from `+/- 10 +/- 10 * i`.
#[inline]
#[must_use]
pub fn random_matrix_2<R: Rng + ?Sized>(rng: &mut R) -> CMatrix2 {
    let d = Uniform::from(-10_f64..10_f64);
    CMatrix2::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)))
}

#[cfg(test)]
mod test {
    use rand::{rngs::StdRng, SeedableRng};
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
        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        let d = Uniform::new(-1_f64, 1_f64);
        let m = CMatrix2::new(
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
        );
        let p = project_to_su2(m);
        assert_eq_matrix!(p, CMatrix2::identity(), EPSILON);
        for _ in 0_u32..100_u32 {
            let r = CMatrix2::from_fn(|_, _| Complex::new(d.sample(&mut rng), d.sample(&mut rng)));
            let p = project_to_su2_unorm(r);
            assert!(p.trace().imaginary().abs() < EPSILON);
            assert!((p * p.adjoint() - CMatrix2::identity() * p.determinant()).norm() < EPSILON);

            assert_matrix_is_su_2!(p / p.determinant().sqrt(), EPSILON);
        }

        for _ in 0_u32..100_u32 {
            let r = CMatrix2::from_fn(|_, _| Complex::new(d.sample(&mut rng), d.sample(&mut rng)));
            let p = project_to_su2(r);
            assert_matrix_is_su_2!(p, EPSILON);
        }
    }

    #[test]
    fn random_su2_t() {
        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        for _ in 0_u32..100_u32 {
            let m = random_su2(&mut rng);
            assert_matrix_is_su_2!(m, EPSILON);
        }
        for _ in 0_u32..100_u32 {
            let m = random_su2(&mut rng);
            assert!(is_matrix_su2(&m, EPSILON));
        }
        for _ in 0_u32..100_u32 {
            let m = random_su2(&mut rng) * Complex::new(1.5_f64, 0.7_f64);
            assert!(!is_matrix_su2(&m, EPSILON));
        }
    }
}
