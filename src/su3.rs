
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
};
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

// u64 is just not enough.
type FactorialNumber = u128;

/// Return N such that `1/(N-7)! < [`f64::EPSILON`].
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

/// size of the factorial storage
const N: usize = 26; // TODO verify accuracy

/// static store for factorial number
struct FactorialStorageStatic {
    data: [FactorialNumber; N]
}

impl FactorialStorageStatic {
    /// compile time evaluation of all 25 factorial numbers
    pub const fn new() -> Self {
        let mut data : [FactorialNumber; N] = [1; N];
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

/// give the SU3 matrix from the adjoint rep, i.e compute `exp(i v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
/// Note that the documentation above explain the algorithm for exp(X) here it is a modified version for
/// exp(i X).
#[inline]
pub fn su3_exp_i(v: Su3Adjoint) -> CMatrix3 {
    // todo optimize even more using f64 to reduce the number of operation using complex that might be useless
    let n = N - 1;
    let mut q0: Complex = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(n).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = v.d();
    let t: Complex = v.t();
    for i in (0..n).rev() {
        let q0_n = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64) + d * q2;
        let q1_n = I * (q0 - t * q2);
        let q2_n = I * q1;
        
        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }
    
    let m = v.to_matrix();
    return CMatrix3::from_diagonal_element(q0) + m * q1 + m * m * q2;
}

/// gives the value `exp(v^a T^a )`
///
/// The algorithm use is much more efficient the diagonalization method.
/// It use the Cayley–Hamilton theorem. If you wish to find more about it you can read the
/// [OpenQCD](https://luscher.web.cern.ch/luscher/openQCD/) documentation that can be found
/// [here](https://github.com/sa2c/OpenQCD-AVX512/blob/master/doc/su3_fcts.pdf) or by downloading a release.
#[inline]
pub fn su3_exp_r(v: Su3Adjoint) -> CMatrix3 {
    let n = N - 1;
    let mut q0: Complex = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(n).unwrap() as f64);
    let mut q1: Complex = Complex::from(0_f64);
    let mut q2: Complex = Complex::from(0_f64);
    let d: Complex = v.d();
    let t: Complex = v.t();
    for i in (0..n).rev() {
        let q0_n = Complex::from(1f64 / *FACTORIAL_STORAGE_STAT.try_get_factorial(i).unwrap() as f64) - I * d * q2;
        let q1_n = q0 - t * q2;
        let q2_n = q1;
        
        q0 = q0_n;
        q1 = q1_n;
        q2 = q2_n;
    }
    
    let m = v.to_matrix();
    return CMatrix3::from_diagonal_element(q0) + m * q1 + m * m * q2;
}

#[cfg(test)]
#[test]
/// test that [`N`] is indeed what we need
fn test_constante(){
    assert_eq!(N, get_factorial_size_for_exp() + 1)
}
