
//! Utils function and structure

use std::convert::TryInto;
use std::ops::{Neg, Mul, MulAssign};
use approx::*;

#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

type FactorialNumber = u128;

/// Smallest number such that (n+1)! overflow u128.
pub const MAX_NUMBER_FACTORIAL: usize = 34;

/// return n!.
///
/// # Panic
/// It overflows if `n >= 35` and panics in debug.
///
/// # Example
/// ```
/// # use lattice_qcd_rs::utils::factorial;
/// assert_eq!(factorial(0), 1);
/// assert_eq!(factorial(4), 24);
/// assert_eq!(factorial(6), 720);
/// assert_eq!(factorial(34), 295232799039604140847618609643520000000);
/// ```
/// ```should_panic
/// # use lattice_qcd_rs::utils::factorial;
/// let n = factorial(34);
/// let (_, overflowed) = n.overflowing_mul(35); // try compute 35! with overflow check.
/// assert!(!overflowed);
/// ```
#[allow(clippy::as_conversions)] // constant function cant use try into
pub const fn factorial(n: usize) -> FactorialNumber {
    if n == 0 {
        1
    }
    else {
        n as FactorialNumber * factorial(n - 1)
    }
}

/// Dynamical size factorial store
pub struct FactorialStorageDyn {
    data: Vec<FactorialNumber>
}

impl FactorialStorageDyn {
    /// Create a new object with an empty storage
    pub const fn new() -> Self {
        Self{data : Vec::new()}
    }
    
    /// Build the storage up to and including `value`.
    ///
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// f.build_storage(6);
    /// assert_eq!(*f.try_get_factorial(6).unwrap(), 720);
    /// ```
    pub fn build_storage(&mut self, value: usize) {
        self.get_factorial(value);
    }
    
    /// Get the factorial number. If it is not already computed build internal storage
    ///
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// assert_eq!(f.get_factorial(6), 720);
    /// assert_eq!(f.get_factorial(4), 24);
    /// ```
    pub fn get_factorial(&mut self, value: usize) -> FactorialNumber {
        let mut len = self.data.len();
        if len == 0 {
            self.data.push(1);
            len = 1;
        }
        if len > value{
            return self.data[value];
        }
        for i in len..value + 1{
            self.data.push(self.data[i - 1] * TryInto::<FactorialNumber>::try_into(i).unwrap());
        }
        self.data[value]
    }
    
    /// try get factorial from storage
    ///
    ///
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// assert_eq!(f.get_factorial(4), 24);
    /// assert_eq!(*f.try_get_factorial(4).unwrap(), 24);
    /// assert_eq!(f.try_get_factorial(6), None);
    /// ```
    pub fn try_get_factorial(&self, value: usize) -> Option<&FactorialNumber> {
        self.data.get(value)
    }
    
    /// Get factorial but does build the storage if it is missing
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// assert_eq!(f.get_factorial(4), 24);
    /// assert_eq!(f.get_factorial_no_storage(6), 720);
    /// assert_eq!(f.try_get_factorial(6), None);
    /// ```
    pub fn get_factorial_no_storage(&self, value: usize) -> FactorialNumber {
        let mut value_m : FactorialNumber = self.data[value.min(self.data.len() -1 )];
        for i in self.data.len() - 1..value{
            value_m *= TryInto::<FactorialNumber>::try_into(i + 1).unwrap();
        }
        value_m
    }
}

/// Represent a sing
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum Sign {
    /// Stricly negative number (non zero)
    Negative,
    /// Stricly positive number ( non zero)
    Positive,
    /// Zero (or very close to zero)
    Zero
}

impl Sign {
    /// return a f64 form the sign `(-1_f64, 0_f64, 1_f64)`.
    pub const fn to_f64(self) -> f64 {
        match self {
            Sign::Negative => -1_f64,
            Sign::Positive => 1_f64,
            Sign::Zero => 0_f64,
        }
    }
    
    /// Get the sign form a f64.
    ///
    /// If the value is very close to zero but not quite the sing will nonetheless be Sign::Zero.
    pub fn sign(f: f64) -> Self {
        // TODO manage NaN
        if relative_eq!(f, 0_f64) {
            Sign::Zero
        }
        else if f > 0_f64 {
            Sign::Positive
        }
        else {
            Sign::Negative
        }
    }
    
    /// Convert the sign to an i8.
    pub const fn to_i8(self) -> i8 {
        match self {
            Sign::Negative => -1_i8,
            Sign::Positive => 1_i8,
            Sign::Zero => 0_i8,
        }
    }
    
}

/// Retuns the sign of `a - b`, witah a and b are usize
pub const fn sign_from_diff(a: usize, b: usize) -> Sign {
    let (result, underflow) = a.overflowing_sub(b);
    match (result, underflow) {
        (0, false) => Sign::Zero,
        (_, true) => Sign::Negative,
        _ => Sign::Positive,
    }
}

impl From<Sign> for f64 {
    fn from(s : Sign) -> f64 {
        s.to_f64()
    }
}

impl From<f64> for Sign {
    fn from(f : f64) -> Sign {
        Sign::sign(f)
    }
}

impl Neg for Sign {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Zero => Sign::Zero,
            Sign::Negative => Sign::Positive,
        }
    }
}

impl Mul for Sign {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Sign::Negative, Sign::Negative) | (Sign::Positive, Sign::Positive) => Sign::Positive,
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (Sign::Positive, Sign::Negative) | (Sign::Negative, Sign::Positive) => Sign::Negative
        }
    }
}

impl MulAssign<Sign> for Sign {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Return the levi civita symbol of the given index
/// # Example
/// ```
/// # use lattice_qcd_rs::utils::{Sign, levi_civita};
/// assert_eq!(Sign::Positive, levi_civita(&[1, 2, 3]));
/// assert_eq!(Sign::Negative, levi_civita(&[2, 1, 3]));
/// ```
pub fn levi_civita(index: &[usize]) -> Sign {
    let mut prod = Sign::Positive;
    for (pos, el_1) in index.iter().enumerate() {
        for el_2 in index.iter().take(pos) {
            prod *= sign_from_diff(*el_1, *el_2);
        }
    }
    prod
}


#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    /// test that the factorial pass for MAX_NUMBER_FACTORIAL
    fn test_factorial_pass() {
        factorial(MAX_NUMBER_FACTORIAL);
    }
    
    #[test]
    #[should_panic]
    #[cfg(not(feature = "no-overflow-test"))]
    /// test that the factorial overflow for MAX_NUMBER_FACTORIAL + 1
    fn test_factorial_bigger() {
        factorial(MAX_NUMBER_FACTORIAL + 1);
    }
    
    #[test]
    #[should_panic]
    /// test that the factorial overflow for MAX_NUMBER_FACTORIAL + 1
    fn test_factorial_overflow() {
        let n = factorial(MAX_NUMBER_FACTORIAL);
        let (_, overflowed) = n.overflowing_mul(MAX_NUMBER_FACTORIAL as u128 + 1);
        assert!(!overflowed);
    }
    
    #[test]
    fn levi_civita_test(){
        assert_eq!(Sign::Positive, levi_civita(&[]));
        assert_eq!(Sign::Positive, levi_civita(&[1, 2]));
        assert_eq!(Sign::Positive, levi_civita(&[0, 1]));
        assert_eq!(Sign::Positive, levi_civita(&[1, 2, 3]));
        assert_eq!(Sign::Positive, levi_civita(&[0, 1, 2]));
        assert_eq!(Sign::Positive, levi_civita(&[3, 1, 2]));
        assert_eq!(Sign::Positive, levi_civita(&[2, 3, 1]));
        assert_eq!(Sign::Positive, levi_civita(&[3, 1, 2, 4]));
        assert_eq!(Sign::Positive, levi_civita(&[1, 3, 4, 2]));
        assert_eq!(Sign::Zero, levi_civita(&[3, 3, 1]));
        assert_eq!(Sign::Zero, levi_civita(&[1, 1, 1]));
        assert_eq!(Sign::Zero, levi_civita(&[1, 1]));
        assert_eq!(Sign::Zero, levi_civita(&[2, 2]));
        assert_eq!(Sign::Negative, levi_civita(&[2, 1]));
        assert_eq!(Sign::Negative, levi_civita(&[1, 0]));
        assert_eq!(Sign::Negative, levi_civita(&[1, 3, 2]));
        assert_eq!(Sign::Negative, levi_civita(&[3, 2, 1]));
        assert_eq!(Sign::Negative, levi_civita(&[2, 1, 3]));
        assert_eq!(Sign::Negative, levi_civita(&[2, 1, 3, 4]));
        
        assert_eq!(Sign::Zero, sign_from_diff(0, 0));
        assert_eq!(Sign::Zero, sign_from_diff(4, 4));
        assert_eq!(Sign::Negative, sign_from_diff(1, 4));
        assert_eq!(Sign::Positive, sign_from_diff(4, 1));
    }
}
