//! Module for SU(3) matrices and su(3) (that is the generators of SU(3) )
//!
//! The module defines the SU(3) generator we use the same matrices as on [wikipedia](https://en.wikipedia.org/w/index.php?title=Gell-Mann_matrices&oldid=988659438#Matrices) **divided by two** such that `Tr(T^a T^b) = \delta^{ab} /2 `.

use na::{base::allocator::Allocator, ComplexField, DefaultAllocator, OMatrix};
use rand_distr::Distribution;

use super::{
    field::Su3Adjoint, su2, utils, utils::FactorialNumber, CMatrix2, CMatrix3, Complex, Real, I,
    ONE, ZERO,
};

/// SU(3) generator
/// ```textrust
/// 0    0.5  0
/// 0.5  0    0
/// 0    0    0
/// ```
pub const GENERATOR_1: CMatrix3 = CMatrix3::new(
    ZERO,
    Complex::new(0.5_f64, 0_f64),
    ZERO,
    // ---
    Complex::new(0.5_f64, 0_f64),
    ZERO,
    ZERO,
    // ---
    ZERO,
    ZERO,
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0   -i/2   0
/// i/2  0     0
/// 0    0     0
/// ```
pub const GENERATOR_2: CMatrix3 = CMatrix3::new(
    ZERO,
    Complex::new(0_f64, -0.5_f64),
    ZERO,
    // ---
    Complex::new(0_f64, 0.5_f64),
    ZERO,
    ZERO,
    // ---
    ZERO,
    ZERO,
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0.5  0    0
/// 0   -0.5  0
/// 0    0    0
/// ```
pub const GENERATOR_3: CMatrix3 = CMatrix3::new(
    Complex::new(0.5_f64, 0_f64),
    ZERO,
    ZERO,
    // ---
    ZERO,
    Complex::new(-0.5_f64, 0_f64),
    ZERO,
    // ---
    ZERO,
    ZERO,
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0    0    0.5
/// 0    0    0
/// 0.5  0    0
/// ```
pub const GENERATOR_4: CMatrix3 = CMatrix3::new(
    ZERO,
    ZERO,
    Complex::new(0.5_f64, 0_f64),
    // ---
    ZERO,
    ZERO,
    ZERO,
    // ---
    Complex::new(0.5_f64, 0_f64),
    ZERO,
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0    0   -i/2
/// 0    0    0
/// i/2  0    0
/// ```
pub const GENERATOR_5: CMatrix3 = CMatrix3::new(
    ZERO,
    ZERO,
    Complex::new(0_f64, -0.5_f64),
    // ---
    ZERO,
    ZERO,
    ZERO,
    // ---
    Complex::new(0_f64, 0.5_f64),
    ZERO,
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0    0    0
/// 0    0    0.5
/// 0    0.5  0
/// ```
pub const GENERATOR_6: CMatrix3 = CMatrix3::new(
    ZERO,
    ZERO,
    ZERO,
    // ---
    ZERO,
    ZERO,
    Complex::new(0.5_f64, 0_f64),
    // ---
    ZERO,
    Complex::new(0.5_f64, 0_f64),
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0    0    0
/// 0    0   -i/2
/// 0    i/2  0
/// ```
pub const GENERATOR_7: CMatrix3 = CMatrix3::new(
    ZERO,
    ZERO,
    ZERO,
    // ---
    ZERO,
    ZERO,
    Complex::new(0_f64, -0.5_f64),
    // ---
    ZERO,
    Complex::new(0_f64, 0.5_f64),
    ZERO,
);

/// SU(3) generator
/// ```textrust
/// 0.5/sqrt(3)  0            0
/// 0            0.5/sqrt(3)  0
/// 0            0           -1/sqrt(3)
/// ```
pub const GENERATOR_8: CMatrix3 = CMatrix3::new(
    Complex::new(ONE_OVER_2_SQRT_3, 0_f64),
    ZERO,
    ZERO,
    // ---
    ZERO,
    Complex::new(ONE_OVER_2_SQRT_3, 0_f64),
    ZERO,
    // ---
    ZERO,
    ZERO,
    Complex::new(MINUS_ONE_OVER_SQRT_3, 0_f64),
);

/// -1/sqrt(3), used for [`GENERATOR_8`].
const MINUS_ONE_OVER_SQRT_3: f64 = -0.577_350_269_189_625_8_f64;
/// 2/sqrt(3), used for [`GENERATOR_8`].
const ONE_OVER_2_SQRT_3: f64 = 0.288_675_134_594_812_9_f64;

/// list of SU(3) generators
/// they are normalize such that `Tr(T^a T^b) = \frac{1}{2}\delta^{ab}`
pub const GENERATORS: [&CMatrix3; 8] = [
    &GENERATOR_1,
    &GENERATOR_2,
    &GENERATOR_3,
    &GENERATOR_4,
    &GENERATOR_5,
    &GENERATOR_6,
    &GENERATOR_7,
    &GENERATOR_8,
];

/// Exponential of matrices.
///
/// Note prefer using [`su3_exp_r`] and [`su3_exp_i`] when possible.
pub trait MatrixExp<T> {
    /// Return the exponential of the matrix
    fn exp(&self) -> T;
}

/// Basic implementation of matrix exponential for complex matrices.
/// It does it by first diagonalizing the matrix then exponentiate the diagonal
/// and retransforms it back to the original basis.
impl<T, D> MatrixExp<OMatrix<T, D, D>> for OMatrix<T, D, D>
where
    T: ComplexField + Copy,
    D: na::DimName + na::DimSub<na::U1>,
    DefaultAllocator: Allocator<T, D, na::DimDiff<D, na::U1>>
        + Allocator<T, na::DimDiff<D, na::U1>>
        + Allocator<T, D, D>
        + Allocator<T, D>,
    OMatrix<T, D, D>: Clone,
{
    fn exp(&self) -> OMatrix<T, D, D> {
        let decomposition = self.clone().schur();
        // a complex matrix is always diagonalisable
        let eigens = decomposition.eigenvalues().unwrap();
        let new_matrix = Self::from_diagonal(&eigens.map(|el| el.exp()));
        let (q, _) = decomposition.unpack();
        // q is always invertible
        q.clone() * new_matrix * q.try_inverse().unwrap()
    }
}

/// Create a [`Matrix3`] of the form `[v1, v2, v1* x v2*]` where v1* is the conjugate of v1 and
/// `x` is the cross product
/// # Example
/// ```ignore
/// # use nalgebra::{Vector3, Complex, Matrix3};
/// # use lattice_qcd_rs::{assert_eq_matrix};
/// let v1 = Vector3::new(
///     Complex::new(1_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
/// );
/// let v2 = Vector3::new(
///     Complex::new(0_f64, 0_f64),
///     Complex::new(1_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
/// );
/// assert_eq_matrix!(
///     Matrix3::identity(),
///     create_matrix_from_2_vector(v1, v2),
///     0.000_000_1_f64
/// );
///
/// let v1 = Vector3::new(
///     Complex::new(0_f64, 1_f64),
///     Complex::new(0_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
/// );
/// let v2 = Vector3::new(
///     Complex::new(0_f64, 0_f64),
///     Complex::new(1_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
/// );
/// let m = Matrix3::new(
///     Complex::new(0_f64, 1_f64),
///     Complex::new(0_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
///     // ---
///     Complex::new(0_f64, 0_f64),
///     Complex::new(1_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
///     // ---
///     Complex::new(0_f64, 0_f64),
///     Complex::new(0_f64, 0_f64),
///     Complex::new(0_f64, -1_f64),
/// );
/// assert_eq_matrix!(
///     m,
///     create_matrix_from_2_vector(v1, v2),
///     0.000_000_1_f64
/// );
/// ```
fn create_matrix_from_2_vector(
    v1: na::Vector3<Complex>,
    v2: na::Vector3<Complex>,
) -> na::Matrix3<Complex> {
    // TODO find a better way
    let cross_vec: na::Vector3<Complex> = v1.conjugate().cross(&v2.conjugate());
    let iter = v1.iter().chain(v2.iter()).chain(cross_vec.iter()).copied();
    na::Matrix3::<Complex>::from_iterator(iter)
}

/// get an orthonormalize matrix from two vector.
fn get_ortho_matrix_from_2_vector(v1: na::Vector3<Complex>, v2: na::Vector3<Complex>) -> CMatrix3 {
    let v1_new = v1.try_normalize(f64::EPSILON).unwrap_or(v1);
    let v2_temp = v2 - v1_new * v1_new.conjugate().dot(&v2);
    let v2_new = v2_temp.try_normalize(f64::EPSILON).unwrap_or(v2_temp);
    create_matrix_from_2_vector(v1_new, v2_new)
}

/// Try orthonormalize the given matrix.
pub fn orthonormalize_matrix(matrix: &CMatrix3) -> CMatrix3 {
    let v1 = na::Vector3::from_iterator(matrix.column(0).iter().copied());
    let v2 = na::Vector3::from_iterator(matrix.column(1).iter().copied());
    get_ortho_matrix_from_2_vector(v1, v2)
}

/// Orthonormalize the given matrix by mutating its content.
pub fn orthonormalize_matrix_mut(matrix: &mut CMatrix3) {
    *matrix = orthonormalize_matrix(matrix);
}

/// Generate Uniformly distributed SU(3) matrix
pub fn get_random_su3(rng: &mut impl rand::Rng) -> CMatrix3 {
    get_rand_su3_with_dis(rng, &rand::distributions::Uniform::new(-1_f64, 1_f64))
}

/// get a random SU3 with the given distribution.
///
/// The given distribution can be quite opaque on the distribution of the SU(3) matrix.
/// For a matrix Uniformly distributed amoung SU(3) use [`get_random_su3`].
/// For a matrix close to unity use [`get_random_su3_close_to_unity`]
fn get_rand_su3_with_dis(
    rng: &mut impl rand::Rng,
    d: &impl rand_distr::Distribution<Real>,
) -> CMatrix3 {
    let mut v1 = get_random_vec_3(rng, d);
    while v1.norm() <= f64::EPSILON {
        v1 = get_random_vec_3(rng, d);
    }
    let mut v2 = get_random_vec_3(rng, d);
    while v1.dot(&v2).modulus() <= f64::EPSILON {
        v2 = get_random_vec_3(rng, d);
    }
    get_ortho_matrix_from_2_vector(v1, v2)
}

/// get a random [`na::Vector3<Complex>`].
fn get_random_vec_3(
    rng: &mut impl rand::Rng,
    d: &impl rand_distr::Distribution<Real>,
) -> na::Vector3<Complex> {
    na::Vector3::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)))
}

/// Get a radom SU(3) matrix close to `[get_r] (+/- 1) * [get_s] (+/- 1) * [get_t] (+/- 1)`.
///
/// Note that it diverges from SU(3) sligthly.
/// `spread_parameter` should be between between 0 and 1 both excluded to generate valide data.
/// outside this boud it will not panic but can have unexpected results.
pub fn get_random_su3_close_to_unity<R>(spread_parameter: Real, rng: &mut R) -> CMatrix3
where
    R: rand::Rng + ?Sized,
{
    let r = get_r(su2::get_random_su2_close_to_unity(spread_parameter, rng));
    let s = get_s(su2::get_random_su2_close_to_unity(spread_parameter, rng));
    let t = get_t(su2::get_random_su2_close_to_unity(spread_parameter, rng));
    let distribution = rand::distributions::Bernoulli::new(0.5_f64).unwrap();
    let mut x = r * s * t;
    if distribution.sample(rng) {
        x = x.adjoint();
    }
    x
}

/// Embed a Matrix2 inside Matrix3 leaving the last row and column be the same as identity.
pub fn get_r(m: CMatrix2) -> CMatrix3 {
    CMatrix3::new(
        m[(0, 0)],
        m[(0, 1)],
        ZERO,
        // ---
        m[(1, 0)],
        m[(1, 1)],
        ZERO,
        // ---
        ZERO,
        ZERO,
        ONE,
    )
}

/// Embed a Matrix2 inside Matrix3 leaving the second row and column be the same as identity.
pub fn get_s(m: CMatrix2) -> CMatrix3 {
    CMatrix3::new(
        m[(0, 0)],
        ZERO,
        m[(0, 1)],
        // ---
        ZERO,
        ONE,
        ZERO,
        // ---
        m[(1, 0)],
        ZERO,
        m[(1, 1)],
    )
}

/// Embed a Matrix2 inside Matrix3 leaving the first row and column be the same as identity.
pub fn get_t(m: CMatrix2) -> CMatrix3 {
    CMatrix3::new(
        ONE,
        ZERO,
        ZERO,
        // ---
        ZERO,
        m[(0, 0)],
        m[(0, 1)],
        // ---
        ZERO,
        m[(1, 0)],
        m[(1, 1)],
    )
}

/// get the [`CMatrix2`] sub block corresponing to [`get_r`]
pub fn get_sub_block_r(m: CMatrix3) -> CMatrix2 {
    CMatrix2::new(
        m[(0, 0)],
        m[(0, 1)],
        // ---
        m[(1, 0)],
        m[(1, 1)],
    )
}

/// get the [`CMatrix2`] sub block corresponing to [`get_s`]
pub fn get_sub_block_s(m: CMatrix3) -> CMatrix2 {
    CMatrix2::new(
        m[(0, 0)],
        m[(0, 2)],
        // ---
        m[(2, 0)],
        m[(2, 2)],
    )
}

/// get the [`CMatrix2`] sub block corresponing to [`get_t`]
pub fn get_sub_block_t(m: CMatrix3) -> CMatrix2 {
    CMatrix2::new(
        m[(1, 1)],
        m[(1, 2)],
        // ---
        m[(2, 1)],
        m[(2, 2)],
    )
}

/// Get the unormalize SU(2) sub matrix of an SU(3) matrix correspondig to the "r" sub block see
/// [`get_sub_block_r`] and [`get_r`].
pub fn get_su2_r_unorm(input: CMatrix3) -> CMatrix2 {
    su2::project_to_su2_unorm(get_sub_block_r(input))
}

/// Get the unormalize SU(2) sub matrix of an SU(3) matrix correspondig to the "s" sub block see
/// [`get_sub_block_s`] and [`get_s`].
pub fn get_su2_s_unorm(input: CMatrix3) -> CMatrix2 {
    su2::project_to_su2_unorm(get_sub_block_s(input))
}

/// Get the unormalize SU(2) sub matrix of an SU(3) matrix correspondig to the "t" sub block see
/// [`get_sub_block_t`] and [`get_t`].
pub fn get_su2_t_unorm(input: CMatrix3) -> CMatrix2 {
    su2::project_to_su2_unorm(get_sub_block_t(input))
}

/// Get the three unormalize sub SU(2) matrix of the given SU(3) matrix, ordered `r, s, t`
pub fn extract_su2_unorm(m: CMatrix3) -> [CMatrix2; 3] {
    [get_su2_r_unorm(m), get_su2_s_unorm(m), get_su2_t_unorm(m)]
}

/// Return the matrix
/// ```textrust
/// U'_{ij} =| -U_{ij} if i not equal to j
///          | U_{ii}  if i = j
/// ```
/// # Example
/// ```
/// # use lattice_qcd_rs::{CMatrix3, su3::reverse, Complex};
/// assert_eq!(CMatrix3::identity(), reverse(CMatrix3::identity()));
/// let m1 = CMatrix3::new(
///     Complex::from(1_f64),
///     Complex::from(2_f64),
///     Complex::from(3_f64),
///     // ---
///     Complex::from(4_f64),
///     Complex::from(5_f64),
///     Complex::from(6_f64),
///     // ---
///     Complex::from(7_f64),
///     Complex::from(8_f64),
///     Complex::from(9_f64),
/// );
/// let m2 = CMatrix3::new(
///     Complex::from(1_f64),
///     -Complex::from(2_f64),
///     -Complex::from(3_f64),
///     // ---
///     -Complex::from(4_f64),
///     Complex::from(5_f64),
///     -Complex::from(6_f64),
///     // ---
///     -Complex::from(7_f64),
///     -Complex::from(8_f64),
///     Complex::from(9_f64),
/// );
/// assert_eq!(m2, reverse(m1));
/// assert_eq!(m1, reverse(m2));
/// ```
pub fn reverse(input: CMatrix3) -> CMatrix3 {
    input.map_with_location(|i, j, el| {
        if i == j {
            el
        }
        else {
            -el
        }
    })
}

/// Return N such that `1/(N-7)!` < [`f64::EPSILON`].
///
/// This number is needed for the computation of exponential matrix
#[allow(clippy::as_conversions)] // no try into for f64
pub fn get_factorial_size_for_exp() -> usize {
    let mut n: usize = 7;
    let mut factorial_value = 1;
    while 1_f64 / (factorial_value as f64) >= Real::EPSILON {
        n += 1;
        factorial_value *= n - 7;
    }
    n
}

/// Size of the array for FactorialStorageStatic
const FACTORIAL_STORAGE_STAT_SIZE: usize = utils::MAX_NUMBER_FACTORIAL + 1;

/// static store for factorial number
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct FactorialStorageStatic {
    data: [FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE],
}

macro_rules! set_factorial_storage {
    ($data:ident, 0) => {
        $data[0] = 1;
    };
    ($data:ident, $e:expr) => {
        $data[$e] = $data[$e - 1] * $e as FactorialNumber;
    };
}

impl FactorialStorageStatic {
    /// compile time evaluation of all 34 factorial numbers
    #[allow(clippy::as_conversions)] // constant function cant use try into
    pub const fn new() -> Self {
        let mut data: [FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE] =
            [1; FACTORIAL_STORAGE_STAT_SIZE];
        let mut i = 1;
        while i < FACTORIAL_STORAGE_STAT_SIZE {
            // still not for loop in const fn
            set_factorial_storage!(data, i);
            i += 1;
        }
        Self { data }
    }

    /// access in O(1). Return None if `value` is bigger than 34
    pub fn try_get_factorial(&self, value: usize) -> Option<&FactorialNumber> {
        self.data.get(value)
    }

    /// Get an iterator over the factorial number form `0!` up to `34!`
    pub fn iter(&self) -> impl Iterator<Item = &FactorialNumber> {
        self.data.iter()
    }

    /// Get the slice of factoral number
    pub const fn as_slice(&self) -> &[FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE] {
        &self.data
    }
}

impl Default for FactorialStorageStatic {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for FactorialStorageStatic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Factorial Storage :")?;
        for (i, n) in self.iter().enumerate() {
            write!(f, "{}! = {},", i, n)?;
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a FactorialStorageStatic {
    type IntoIter = <&'a [u128; FACTORIAL_STORAGE_STAT_SIZE] as IntoIterator>::IntoIter;
    type Item = &'a u128;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl AsRef<[FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE]> for FactorialStorageStatic {
    fn as_ref(&self) -> &[FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE] {
        self.as_slice()
    }
}

/// factorial number storage in order to find the exponential in O(1) for a set storage
/// the set if for all number `N` such that `\frac{1}{(N-7)!} >= \mathrm{f64::EPSILON}`
const FACTORIAL_STORAGE_STAT: FactorialStorageStatic = FactorialStorageStatic::new();

/// number of step for the computation of matric exponential using the Cayley–Hamilton theorem.
const N: usize = 26;

/// give the SU3 matrix from the adjoint rep, i.e compute `exp(i v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
/// Note that the documentation above explain the algorithm for exp(X) here it is a modified version for
/// exp(i X).
#[inline]
#[allow(clippy::as_conversions)] // no try into for f64
pub fn su3_exp_i(su3_adj: Su3Adjoint) -> CMatrix3 {
    // todo optimize even more using f64 to reduce the number of operation using complex that might be useless
    const N_LOOP: usize = N - 1;
    let mut q0: Complex =
        Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(N_LOOP).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = su3_adj.d();
    let t: Complex = su3_adj.t().into();
    for i in (0..N_LOOP).rev() {
        let q0_n =
            Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64)
                + d * q2;
        let q1_n = I * (q0 - t * q2);
        let q2_n = I * q1;

        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }

    let m = su3_adj.to_matrix();
    CMatrix3::from_diagonal_element(q0) + m * q1 + m * m * q2
}

/// the input must be a su(3) matrix (generator of SU(3)), gives the SU3 matrix from the adjoint rep,
/// i.e compute `exp(i v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
/// Note that the documentation above explain the algorithm for exp(X) here it is a modified version for
/// exp(i X).
///
/// # Panic
/// The input matrix must be an su(3) (Lie algebra of SU(3)) matrix or approximatively su(3),
/// otherwise the function will panic in debug mod, in release the ouptut gives unexpected values.
/// ```should_panic
/// use lattice_qcd_rs::{
///     assert_eq_matrix,
///     su3::{matrix_su3_exp_i, MatrixExp},
/// };
/// use nalgebra::{Complex, Matrix3};
/// let i = Complex::new(0_f64, 1_f64);
/// let matrix = Matrix3::identity(); // this is NOT an su(3) matrix
/// let output = matrix_su3_exp_i(matrix);
/// // We panic in debug. In release the following asset will fail.
/// // assert_eq_matrix!(output, (matrix* i).exp(), f64::EPSILON * 100_000_f64);
/// ```
#[inline]
#[allow(clippy::as_conversions)] // no try into for f64
pub fn matrix_su3_exp_i(matrix: CMatrix3) -> CMatrix3 {
    debug_assert!(is_matrix_su3_lie(&matrix, f64::EPSILON * 100_f64));
    const N_LOOP: usize = N - 1;
    let mut q0: Complex =
        Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(N_LOOP).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = matrix.determinant() * I;
    let t: Complex = -Complex::from(0.5_f64) * (matrix * matrix).trace();
    for i in (0..N_LOOP).rev() {
        let q0_n =
            Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64)
                + d * q2;
        let q1_n = I * (q0 - t * q2);
        let q2_n = I * q1;

        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }

    CMatrix3::from_diagonal_element(q0) + matrix * q1 + matrix * matrix * q2
}

/// gives the value `exp(v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
#[inline]
#[allow(clippy::as_conversions)] // no try into for f64
pub fn su3_exp_r(su3_adj: Su3Adjoint) -> CMatrix3 {
    const N_LOOP: usize = N - 1;
    let mut q0: Complex =
        Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(N_LOOP).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = su3_adj.d();
    let t: Complex = su3_adj.t().into();
    for i in (0..N_LOOP).rev() {
        let q0_n =
            Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64)
                - I * d * q2;
        let q1_n = q0 - t * q2;
        let q2_n = q1;

        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }

    let m = su3_adj.to_matrix();
    CMatrix3::from_diagonal_element(q0) + m * q1 + m * m * q2
}

/// the input must be a su(3) matrix (generator of SU(3)), gives the value `exp(v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
///
/// # Panic
/// The input matrix must be an su(3) (Lie algebra of SU(3)) matrix or approximatively su(3),
/// otherwise the function will panic in debug mod, in release the ouptut gives unexpected values.
/// ```should_panic
/// use lattice_qcd_rs::{
///     assert_eq_matrix,
///     su3::{matrix_su3_exp_r, MatrixExp},
/// };
/// use nalgebra::{Complex, Matrix3};
/// let i = Complex::new(0_f64, 1_f64);
/// let matrix = Matrix3::identity(); // this is NOT an su(3)
/// let output = matrix_su3_exp_r(matrix);
/// // We panic in debug. In release the following asset will fail.
/// // assert_eq_matrix!(output, matrix.exp(), f64::EPSILON * 100_000_f64);
/// ```
#[inline]
#[allow(clippy::as_conversions)] // no try into for f64
pub fn matrix_su3_exp_r(matrix: CMatrix3) -> CMatrix3 {
    debug_assert!(is_matrix_su3_lie(&matrix, f64::EPSILON * 100_f64));
    const N_LOOP: usize = N - 1;
    let mut q0: Complex =
        Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(N_LOOP).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = matrix.determinant() * I;
    let t: Complex = -Complex::from(0.5_f64) * (matrix * matrix).trace();
    for i in (0..N_LOOP).rev() {
        let q0_n =
            Complex::from(1_f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64)
                - I * d * q2;
        let q1_n = q0 - t * q2;
        let q2_n = q1;

        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }

    CMatrix3::from_diagonal_element(q0) + matrix * q1 + matrix * matrix * q2
}

/// Return wether the input matrix is SU(3) up to epsilon.
pub fn is_matrix_su3(m: &CMatrix3, epsilon: f64) -> bool {
    ((m.determinant() - Complex::from(1_f64)).modulus_squared() < epsilon)
        && ((m * m.adjoint() - CMatrix3::identity()).norm() < epsilon)
}

/// Returns wether the given matric is in the lie algebra su(3) that generates SU(3) up to epsilon.
pub fn is_matrix_su3_lie(matrix: &CMatrix3, epsilon: Real) -> bool {
    matrix.trace().modulus() < epsilon && (matrix - matrix.adjoint()).norm() < epsilon
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;

    use super::*;
    use crate::{assert_eq_matrix, I};

    const EPSILON: f64 = 0.000_000_001_f64;
    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    /// test that [`N`] is indeed what we need
    fn test_constante() {
        assert_eq!(N, get_factorial_size_for_exp() + 1)
    }

    #[test]
    fn test_factorial() {
        for i in 0..FACTORIAL_STORAGE_STAT_SIZE {
            assert_eq!(
                *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap(),
                utils::factorial(i)
            );
        }
    }

    #[test]
    fn sub_block() {
        let m = CMatrix3::new(
            Complex::from(1_f64),
            Complex::from(2_f64),
            Complex::from(3_f64),
            // ---
            Complex::from(4_f64),
            Complex::from(5_f64),
            Complex::from(6_f64),
            // ---
            Complex::from(7_f64),
            Complex::from(8_f64),
            Complex::from(9_f64),
        );
        let r = CMatrix2::new(
            Complex::from(1_f64),
            Complex::from(2_f64),
            // ---
            Complex::from(4_f64),
            Complex::from(5_f64),
        );
        assert_eq_matrix!(get_sub_block_r(m), r, EPSILON);

        let m_r = CMatrix3::new(
            Complex::from(1_f64),
            Complex::from(2_f64),
            Complex::from(0_f64),
            // ---
            Complex::from(4_f64),
            Complex::from(5_f64),
            Complex::from(0_f64),
            // ---
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(1_f64),
        );
        assert_eq_matrix!(get_r(r), m_r, EPSILON);

        let s = CMatrix2::new(
            Complex::from(1_f64),
            Complex::from(3_f64),
            // ---
            Complex::from(7_f64),
            Complex::from(9_f64),
        );
        assert_eq_matrix!(get_sub_block_s(m), s, EPSILON);

        let m_s = CMatrix3::new(
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(3_f64),
            // ---
            Complex::from(0_f64),
            Complex::from(1_f64),
            Complex::from(0_f64),
            // ---
            Complex::from(7_f64),
            Complex::from(0_f64),
            Complex::from(9_f64),
        );
        assert_eq_matrix!(get_s(s), m_s, EPSILON);

        let t = CMatrix2::new(
            Complex::from(5_f64),
            Complex::from(6_f64),
            // ---
            Complex::from(8_f64),
            Complex::from(9_f64),
        );
        assert_eq_matrix!(get_sub_block_t(m), t, EPSILON);

        let m_t = CMatrix3::new(
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            // ---
            Complex::from(0_f64),
            Complex::from(5_f64),
            Complex::from(6_f64),
            // ---
            Complex::from(0_f64),
            Complex::from(8_f64),
            Complex::from(9_f64),
        );
        assert_eq_matrix!(get_t(t), m_t, EPSILON);

        for p in &extract_su2_unorm(m) {
            assert_matrix_is_su_2!(p / p.determinant().sqrt(), EPSILON);
        }
    }

    #[test]
    fn exp_matrix() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let d = rand::distributions::Uniform::from(-1_f64..1_f64);
        for m in &GENERATORS {
            let output_r = matrix_su3_exp_r(**m);
            assert_eq_matrix!(output_r, m.exp(), EPSILON);
            let output_i = matrix_su3_exp_i(**m);
            assert_eq_matrix!(output_i, (*m * I).exp(), EPSILON);
        }
        for _ in 0..100 {
            let v = Su3Adjoint::random(&mut rng, &d);
            let m = v.to_matrix();
            let output_r = matrix_su3_exp_r(m);
            assert_eq_matrix!(output_r, m.exp(), EPSILON);
            let output_i = matrix_su3_exp_i(m);
            assert_eq_matrix!(output_i, (m * I).exp(), EPSILON);
        }
    }
    #[test]
    fn test_is_matrix_su3_lie() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let d = rand::distributions::Uniform::from(-1_f64..1_f64);
        for m in &GENERATORS {
            assert!(is_matrix_su3_lie(*m, f64::EPSILON * 100_f64));
        }
        for _ in 0..100 {
            let v = Su3Adjoint::random(&mut rng, &d);
            let m = v.to_matrix();
            assert!(is_matrix_su3_lie(&m, f64::EPSILON * 100_f64));
        }
    }

    #[test]
    fn test_gen() {
        assert_eq!(
            CMatrix3::new(
                Complex::new(1_f64, 0_f64),
                ZERO,
                ZERO,
                // ---
                ZERO,
                Complex::new(1_f64, 0_f64),
                ZERO,
                // ---
                ZERO,
                ZERO,
                Complex::new(-2_f64, 0_f64),
            ) * Complex::from(0.5_f64 / 3_f64.sqrt()),
            GENERATOR_8
        );
    }

    #[test]
    fn test_is_matrix_su3() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let m = CMatrix3::new(
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
        );
        assert!(!is_matrix_su3(&m, f64::EPSILON * 100_f64));
        let m = CMatrix3::new(
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
            Complex::from(0_f64),
        );
        assert!(!is_matrix_su3(&m, f64::EPSILON * 100_f64),);
        let m = CMatrix3::new(
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(1_f64),
            Complex::from(0_f64),
            Complex::from(1_f64),
        );
        assert!(!is_matrix_su3(&m, f64::EPSILON * 100_f64));
        for _ in 0..10 {
            let m = get_random_su3(&mut rng);
            assert!(is_matrix_su3(&m, f64::EPSILON * 100_f64));
        }
        for _ in 0..10 {
            let m = get_random_su3(&mut rng) * Complex::from(1.1_f64);
            assert!(!is_matrix_su3(&m, f64::EPSILON * 100_f64));
        }
    }

    #[test]
    fn factorial_storage() {
        let storage = FactorialStorageStatic::new();
        for (i, f) in storage.data.iter().enumerate() {
            assert_eq!(utils::factorial(i), *f);
        }
        assert!(storage.try_get_factorial(35).is_none());
        assert_eq!(storage.try_get_factorial(34), Some(&utils::factorial(34)));
    }
}
