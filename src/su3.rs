
//! Module for SU(3) matrices and su(3) (that is the generators of SU(3) )
//!
//! The module defines the SU(3) generator we use the same matrices as on [wikipedia](https://en.wikipedia.org/w/index.php?title=Gell-Mann_matrices&oldid=988659438#Matrices) **divided by two** such that `Tr(T^a T^b) = \delta^{ab} /2 `.

use na::{
    ComplexField,
    MatrixN,
    base::allocator::Allocator,
    DefaultAllocator,
};
use super::{
    Complex,
    ONE,
    I,
    ZERO,
    CMatrix3,
    field::Su3Adjoint,
    Real,
    utils,
    CMatrix2,
    su2,
};
use rand_distr::Distribution;
use once_cell::sync::Lazy;

/// SU(3) generator
/// ```math
/// 0    0.5  0
/// 0.5  0    0
/// 0    0    0
/// ```
pub static GENERATOR_1: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ONE, ZERO,
        ONE, ZERO, ZERO,
        ZERO, ZERO, ZERO,
) * Complex::from(0.5_f64));

/// SU(3) generator
/// ```math
/// 0   -i/2   0
/// i/2  0     0
/// 0    0     0
/// ```
pub static GENERATOR_2: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, -I, ZERO,
        I, ZERO, ZERO,
        ZERO, ZERO, ZERO,
) * Complex::from(0.5_f64));

/// SU(3) generator
/// ```math
/// 0.5  0    0
/// 0   -0.5  0
/// 0    0    0
/// ```
pub static GENERATOR_3: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ONE, ZERO, ZERO,
        ZERO, -ONE, ZERO,
        ZERO, ZERO, ZERO,
) * Complex::from(0.5_f64));

/// SU(3) generator
/// ```math
/// 0    0    0.5
/// 0    0    0
/// 0.5  0    0
/// ```
pub static GENERATOR_4: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, ONE,
        ZERO, ZERO, ZERO,
        ONE, ZERO, ZERO,
) * Complex::from(0.5_f64) );

/// SU(3) generator
/// ```math
/// 0    0   -i/2
/// 0    0    0
/// i/2  0    0
/// ```
pub static GENERATOR_5: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, -I,
        ZERO, ZERO, ZERO,
        I, ZERO, ZERO,
) * Complex::from(0.5_f64));

/// SU(3) generator
/// ```math
/// 0    0    0
/// 0    0    0.5
/// 0    0.5  0
/// ```
pub static GENERATOR_6: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, ZERO,
        ZERO, ZERO, ONE,
        ZERO, ONE, ZERO,
) * Complex::from(0.5_f64));

/// SU(3) generator
/// ```math
/// 0    0    0
/// 0    0   -i/2
/// 0    i/2  0
/// ```
pub static GENERATOR_7: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, ZERO,
        ZERO, ZERO, -I,
        ZERO, I, ZERO,
) * Complex::from(0.5_f64));

/// SU(3) generator
/// ```math
/// 0.5/sqrt(3)  0            0
/// 0            0.5/sqrt(3)  0
/// 0            0           -1/sqrt(3)
/// ```
pub static GENERATOR_8: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        Complex::new(1_f64, 0_f64), ZERO, ZERO,
        ZERO, Complex::new(1_f64, 0_f64), ZERO,
        ZERO, ZERO, Complex::new(-2_f64, 0_f64),
) * Complex::from(0.5_f64 / 3_f64.sqrt()));

/// list of SU(3) generators
/// they are normalize such that `Tr(T^a T^b) = \frac{1}{2}\delta^{ab}`
pub static GENERATORS: Lazy<[&CMatrix3; 8]> = Lazy::new(||
    [&GENERATOR_1, &GENERATOR_2, &GENERATOR_3, &GENERATOR_4, &GENERATOR_5, &GENERATOR_6, &GENERATOR_7, &GENERATOR_8]
);


/// Exponential of matrices.
///
/// Note prefer using [`su3_exp_r`] and [`su3_exp_i`] when possible.
pub trait MatrixExp<T> {
    fn exp(&self) -> T;
}

/// Basic implementation of matrix exponential for complex matrices.
/// It does it by first diagonalizing the matrix then exponentiate the diagonal
/// and retransforms it back to the original basis.
impl<T, D> MatrixExp<MatrixN<T, D>> for MatrixN<T, D>
    where T: ComplexField + Copy,
    D: na::DimName + na::DimSub<na::U1>,
    DefaultAllocator: Allocator<T, D, na::DimDiff<D, na::U1>>
        + Allocator<T, na::DimDiff<D, na::U1>>
        + Allocator<T, D, D>
        + Allocator<T, D>,
    MatrixN<T, D> : Clone,
{
    fn exp(&self) -> MatrixN<T, D> {
        let decomposition = self.clone().schur();
        // a complex matrix is always diagonalisable
        let eigens = decomposition.eigenvalues().unwrap();
        let new_matrix = Self::from_diagonal(&eigens.map(|el| el.exp()));
        let (q, _) = decomposition.unpack();
        // q is always invertible
        return q.clone() * new_matrix * q.try_inverse().unwrap();
    }
    
}

/// Create a matrix (v1, v2 , v1* x v2*)
fn create_matrix_from_2_vector(v1: na::Vector3<Complex>, v2:  na::Vector3<Complex>) -> na::Matrix3<Complex> {
    // TODO find a better way
    let cross_vec: na::Vector3<Complex> = v1.conjugate().cross(&v2.conjugate());
    let iter = v1.iter().chain(v2.iter()).chain(cross_vec.iter()).copied();
    na::Matrix3::<Complex>::from_iterator(iter)
}

/// get an orthonormalize matrix from two vector.
fn get_ortho_matrix_from_2_vector(v1: na::Vector3<Complex>, v2: na::Vector3<Complex>) -> CMatrix3 {
    // TODO clean up
    let v1_new = v1.try_normalize(f64::EPSILON).unwrap_or(v1);
    let v2_temp = v2 - v1_new * v1_new.conjugate().dot(&v2);
    let v2_new = v2_temp.try_normalize(f64::EPSILON).unwrap_or(v2_temp);
    create_matrix_from_2_vector(v1_new, v2_new)
}

/// Try orthonormalize the given matrix.
pub fn orthonormalize_matrix(matrix: &CMatrix3) -> CMatrix3 {
    // TODO clean up
    let v1 = na::Vector3::from_iterator(matrix.column(0).iter().copied());
    let v2 = na::Vector3::from_iterator(matrix.column(1).iter().copied());
    get_ortho_matrix_from_2_vector(v1, v2)
}

/// Orthonormalize the given matrix by mutating its content.
pub fn orthonormalize_matrix_mut(matrix: &mut CMatrix3) {
    *matrix = orthonormalize_matrix(matrix);
}

/// Generate Uniformly distributed SU(3)
pub fn get_random_su3(rng: &mut impl rand::Rng) -> CMatrix3 {
    get_rand_su3_with_dis(rng, &rand::distributions::Uniform::new(-1_f64, 1_f64))
}

fn get_rand_su3_with_dis(rng: &mut impl rand::Rng, d: &impl rand_distr::Distribution<Real>) -> CMatrix3 {
    let mut v1 = get_random_vec_3(rng, d);
    while v1.norm() == 0_f64 {
        v1 = get_random_vec_3(rng, d);
    }
    let mut v2 = get_random_vec_3(rng, d);
    while v1.dot(&v2) == Complex::from(0_f64) {
        v2 = get_random_vec_3(rng, d);
    }
    get_ortho_matrix_from_2_vector(v1, v2)
}

/// get a random Vec 3.
fn get_random_vec_3 (rng: &mut impl rand::Rng, d: &impl rand_distr::Distribution<Real>) -> na::Vector3<Complex> {
    na::Vector3::from_fn(|_, _| Complex::new(d.sample(rng), d.sample(rng)))
}

/// Get a radom SU(3) matrix close the origine.
///
/// Note that it diverge from SU(3) sligthly.
pub fn get_random_su3_close_to_unity(spread_parameter: Real, rng: &mut impl rand::Rng) -> CMatrix3 {
    let mut r = get_r(su2::get_random_su2_close_to_unity(spread_parameter, rng));
    let mut s = get_s(su2::get_random_su2_close_to_unity(spread_parameter, rng));
    let mut t = get_t(su2::get_random_su2_close_to_unity(spread_parameter, rng));
    let d = rand::distributions::Bernoulli::new(0.5).unwrap();
    if d.sample(rng) {
        r = r.adjoint();
    }
    if d.sample(rng) {
        s = s.adjoint();
    }
    if d.sample(rng) {
        t = t.adjoint();
    }
    r * s * t
}

fn get_r (m: CMatrix2) -> CMatrix3 {
    CMatrix3::new(
        m[(0,0)], m[(0,1)], ZERO,
        m[(1,0)], m[(1,1)], ZERO,
        ZERO, ZERO, ONE,
    )
}

fn get_s (m: CMatrix2) -> CMatrix3 {
    CMatrix3::new(
        m[(0,0)], ZERO, m[(0,1)],
        ZERO, ONE, ZERO,
        m[(1,0)], ZERO, m[(1,1)],
    )
}

fn get_t (m: CMatrix2) -> CMatrix3 {
    CMatrix3::new(
        ONE, ZERO, ZERO,
        ZERO, m[(0,0)], m[(0,1)],
        ZERO, m[(1,0)], m[(1,1)],
    )
}

// u64 is just not enough.
type FactorialNumber = u128;

/// Return N such that `1/(N-7)!` < [`f64::EPSILON`].
///
/// This number is needed for the computation of exponential matrix
pub fn get_factorial_size_for_exp() -> usize {
    let mut n : usize = 7;
    let mut factorial_value = 1;
    while 1_f64 / (factorial_value as Real) >= Real::EPSILON {
        n += 1;
        factorial_value *= n - 7;
    }
    return n;
}

const FACTORIAL_STORAGE_STAT_SIZE: usize =  utils::MAX_NUMBER_FACTORIAL + 1;

/// static store for factorial number
struct FactorialStorageStatic {
    data: [FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE]
}

impl FactorialStorageStatic {
    /// compile time evaluation of all 25 factorial numbers
    pub const fn new() -> Self {
        let mut data : [FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE] = [1; FACTORIAL_STORAGE_STAT_SIZE];
        // cant do for in constant function :(
        let i = 1;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 2;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 3;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 4;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 5;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 6;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 7;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 8;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 9;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 10;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 11;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 12;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 13;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 14;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 15;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 16;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 17;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 18;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 19;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 20;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 21;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 22;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 23;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 24;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 25;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 26;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 27;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 28;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 29;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 30;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 31;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 32;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 33;
        data[i] = data[i - 1] * i as FactorialNumber;
        let i = 34;
        data[i] = data[i - 1] * i as FactorialNumber;
        Self {data}
    }
    
    /// access in O(1). Return None if `value` is bigger than 25
    pub fn try_get_factorial(&self, value: usize) -> Option<&FactorialNumber> {
        self.data.get(value)
    }
}

/// factorial number storage in order to find the exponential in O(1) for a set storage
/// the set if for all number `N` such that `\frac{1}{(N-7)!} >= \mathrm{f64::EPSILON}`
const FACTORIAL_STORAGE_STAT : FactorialStorageStatic = FactorialStorageStatic::new();

/// size of the factorial storage
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
pub fn su3_exp_i(su3_adj: Su3Adjoint) -> CMatrix3 {
    // todo optimize even more using f64 to reduce the number of operation using complex that might be useless
    let n = N - 1;
    let mut q0: Complex = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(n).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = su3_adj.d();
    let t: Complex = su3_adj.t();
    for i in (0..n).rev() {
        let q0_n = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64) + d * q2;
        let q1_n = I * (q0 - t * q2);
        let q2_n = I * q1;
        
        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }
    
    let m = su3_adj.to_matrix();
    CMatrix3::from_diagonal_element(q0) + m * q1 + m * m * q2
}

/// gives the value `exp(v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
#[inline]
pub fn su3_exp_r(su3_adj: Su3Adjoint) -> CMatrix3 {
    let n = N - 1;
    let mut q0: Complex = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(n).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = su3_adj.d();
    let t: Complex = su3_adj.t();
    for i in (0..n).rev() {
        let q0_n = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64) - I * d * q2;
        let q1_n = q0 - t * q2;
        let q2_n = q1;
        
        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }
    
    let m = su3_adj.to_matrix();
    CMatrix3::from_diagonal_element(q0) + m * q1 + m * m * q2
}

#[cfg(test)]
#[test]
/// test that [`N`] is indeed what we need
fn test_constante(){
    assert_eq!(N, get_factorial_size_for_exp() + 1)
}
