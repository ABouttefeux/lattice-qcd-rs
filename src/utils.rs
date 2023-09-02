//! Utils functions and structures.
//!
//! Mainly things that I do not know where to put.

use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt::{self, Display};
use std::ops::{Mul, MulAssign, Neg};

use approx::abs_diff_eq;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

pub(crate) type FactorialNumber = u128;

/// Smallest number such that `(n+1)!` overflow [`u128`].
pub const MAX_NUMBER_FACTORIAL: usize = 34;

/// return n! (n factorial).
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
#[inline]
#[must_use]
pub const fn factorial(n: usize) -> FactorialNumber {
    if n == 0 {
        1
    } else {
        n as FactorialNumber * factorial(n - 1)
    }
}

/// Dynamical size factorial storage.
///
/// Used as a lazy cache for factorial number. This is not actually used and might be removed later.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FactorialStorageDyn {
    data: Vec<FactorialNumber>,
}

impl Default for FactorialStorageDyn {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl FactorialStorageDyn {
    /// Create a new object with an empty storage.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Build the storage up to and including `value`.
    ///
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// f.build_storage(6);
    /// assert_eq!(*f.try_factorial(6).unwrap(), 720);
    /// ```
    #[allow(clippy::let_underscore_must_use)]
    #[inline]
    pub fn build_storage(&mut self, value: usize) {
        let _: FactorialNumber = self.factorial(value);
    }

    /// Get the factorial number. If it is not already computed build internal storage
    ///
    /// # Panic
    /// panic if value is greater than [`MAX_NUMBER_FACTORIAL`] (34) in debug, overflows otherwise.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// assert_eq!(f.factorial(6), 720);
    /// assert_eq!(f.factorial(4), 24);
    /// ```
    #[allow(clippy::missing_panics_doc)] // never panics
    #[inline]
    #[must_use]
    pub fn factorial(&mut self, value: usize) -> FactorialNumber {
        let mut len = self.data.len();
        if len == 0 {
            self.data.push(1);
            len = 1;
        }
        if len > value {
            return self.data[value];
        }
        for i in len..=value {
            self.data.push(
                self.data[i - 1]
                    * TryInto::<FactorialNumber>::try_into(i).expect("conversion always possible"),
            );
        }
        self.data[value]
    }

    /// try get factorial from storage
    ///
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// assert_eq!(f.factorial(4), 24);
    /// assert_eq!(*f.try_factorial(4).unwrap(), 24);
    /// assert_eq!(f.try_factorial(6), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn try_factorial(&self, value: usize) -> Option<&FactorialNumber> {
        self.data.get(value)
    }

    /// Get factorial but does build the storage if it is missing
    /// #Example
    /// ```
    /// # use lattice_qcd_rs::utils::FactorialStorageDyn;
    /// let mut f = FactorialStorageDyn::new();
    /// assert_eq!(f.factorial(4), 24);
    /// assert_eq!(f.factorial_no_storage(6), 720);
    /// assert_eq!(f.try_factorial(6), None);
    /// ```
    #[allow(clippy::missing_panics_doc)] // never panics
    #[inline]
    #[must_use]
    pub fn factorial_no_storage(&self, value: usize) -> FactorialNumber {
        let mut value_m = self.data[value.min(self.data.len() - 1)];
        for i in self.data.len() - 1..value {
            value_m *=
                TryInto::<FactorialNumber>::try_into(i + 1).expect("conversion always possible");
        }
        value_m
    }
}

/// Represent a sign.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy, Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[allow(clippy::exhaustive_enums)]
pub enum Sign {
    /// Strictly negative number (non zero)
    Negative,
    /// Strictly positive number ( non zero)
    Positive,
    /// Zero (or very close to zero)
    #[default]
    Zero,
}

impl Sign {
    /// return a f64 form the sign `(-1_f64, 0_f64, 1_f64)`.
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        match self {
            Self::Negative => -1_f64,
            Self::Positive => 1_f64,
            Self::Zero => 0_f64,
        }
    }

    /// Get the sign form a f64.
    ///
    /// If the value is very close to zero but not quite the sing will nonetheless be [`Sign::Zero`].
    /// If f is NaN the sing will be [`Sign::Zero`].
    #[inline]
    #[must_use]
    pub fn sign_f64(f: f64) -> Self {
        if abs_diff_eq!(f, 0_f64) || f.is_nan() {
            Self::Zero
        } else if f > 0_f64 {
            Self::Positive
        } else {
            Self::Negative
        }
    }

    /// Convert the sign to an i8.
    #[inline]
    #[must_use]
    pub const fn to_i8(self) -> i8 {
        match self {
            Self::Negative => -1_i8,
            Self::Positive => 1_i8,
            Self::Zero => 0_i8,
        }
    }

    /// Get the sign of the given [`i8`]
    #[inline]
    #[must_use]
    pub const fn sign_i8(n: i8) -> Self {
        if n == 0 {
            Self::Zero
        } else if n > 0 {
            Self::Positive
        } else {
            Self::Negative
        }
    }

    /// Returns the sign of `a - b`, where `a` and `b` are usize
    #[inline]
    #[must_use]
    pub const fn sign_from_diff(a: usize, b: usize) -> Self {
        if a == b {
            Self::Zero
        } else if a > b {
            Self::Positive
        } else {
            Self::Negative
        }
    }
}

impl Display for Sign {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positive => write!(f, "positive"),
            Self::Zero => write!(f, "zero"),
            Self::Negative => write!(f, "negative"),
        }
    }
}

impl From<Sign> for f64 {
    #[inline]
    fn from(s: Sign) -> Self {
        s.to_f64()
    }
}

impl From<f64> for Sign {
    #[inline]
    fn from(f: f64) -> Self {
        Self::sign_f64(f)
    }
}

impl From<Sign> for i8 {
    #[inline]
    fn from(s: Sign) -> Self {
        s.to_i8()
    }
}

impl From<i8> for Sign {
    #[inline]
    fn from(i: i8) -> Self {
        Self::sign_i8(i)
    }
}

impl Neg for Sign {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Self::Positive => Self::Negative,
            Self::Zero => Self::Zero,
            Self::Negative => Self::Positive,
        }
    }
}

impl Mul for Sign {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Negative, Self::Negative) | (Self::Positive, Self::Positive) => Self::Positive,
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (Self::Positive, Self::Negative) | (Self::Negative, Self::Positive) => Self::Negative,
        }
    }
}

impl MulAssign<Self> for Sign {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl PartialOrd for Sign {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Sign {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_i8().cmp(&other.to_i8())
    }
}

/// Return the levi civita symbol of the given index
/// # Example
/// ```
/// # use lattice_qcd_rs::utils::{Sign, levi_civita};
/// assert_eq!(Sign::Positive, levi_civita(&[1, 2, 3]));
/// assert_eq!(Sign::Negative, levi_civita(&[2, 1, 3]));
/// assert_eq!(Sign::Zero, levi_civita(&[2, 2, 3]));
/// ```
#[inline]
#[must_use]
pub const fn levi_civita(index: &[usize]) -> Sign {
    let mut prod = 1_i8;
    let mut i = 0_usize;
    while i < index.len() {
        let mut j = 0_usize;
        while j < i {
            prod *= Sign::sign_from_diff(index[i], index[j]).to_i8();
            j += 1;
        }
        i += 1;
    }
    Sign::sign_i8(prod)
}

#[cfg(test)]
mod test {
    use super::*;

    /// test that the factorial pass for [`MAX_NUMBER_FACTORIAL`]
    #[allow(clippy::missing_const_for_fn)]
    #[allow(clippy::let_underscore_must_use)]
    #[test]
    fn test_factorial_pass() {
        let _: FactorialNumber = factorial(MAX_NUMBER_FACTORIAL);
    }

    #[allow(clippy::missing_const_for_fn)]
    #[allow(clippy::let_underscore_must_use)]
    #[test]
    #[should_panic]
    #[cfg(feature = "overflow-test")]
    /// test that the factorial overflow for `MAX_NUMBER_FACTORIAL + 1`
    fn test_factorial_bigger() {
        let _: FactorialNumber = factorial(MAX_NUMBER_FACTORIAL + 1);
    }

    #[test]
    #[should_panic]
    /// test that the factorial overflow for `MAX_NUMBER_FACTORIAL + 1`
    fn test_factorial_overflow() {
        let n = factorial(MAX_NUMBER_FACTORIAL);
        let (_, overflowed) = n.overflowing_mul(MAX_NUMBER_FACTORIAL as u128 + 1);
        assert!(!overflowed);
    }

    #[test]
    fn sign_i8() {
        assert_eq!(Sign::sign_i8(0), Sign::Zero);
        assert_eq!(Sign::sign_i8(-1), Sign::Negative);
        assert_eq!(Sign::sign_i8(1), Sign::Positive);
        assert_eq!(0, Sign::Zero.to_i8());
        assert_eq!(-1, Sign::Negative.to_i8());
        assert_eq!(1, Sign::Positive.to_i8());
    }

    #[test]
    fn levi_civita_test() {
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

        assert_eq!(Sign::Zero, Sign::sign_from_diff(0, 0));
        assert_eq!(Sign::Zero, Sign::sign_from_diff(4, 4));
        assert_eq!(Sign::Negative, Sign::sign_from_diff(1, 4));
        assert_eq!(Sign::Positive, Sign::sign_from_diff(4, 1));
    }

    #[allow(clippy::cognitive_complexity)]
    #[allow(clippy::float_cmp)]
    #[test]
    fn sign() {
        assert_eq!(Sign::sign_f64(0_f64).to_f64(), 0_f64);
        assert_eq!(Sign::sign_f64(1_f64).to_f64(), 1_f64);
        assert_eq!(Sign::sign_f64(-1_f64).to_f64(), -1_f64);
        assert_eq!(Sign::sign_f64(34_f64), Sign::Positive);
        assert_eq!(Sign::sign_f64(-34_f64), Sign::Negative);
        assert_eq!(Sign::from(-34_f64), Sign::Negative);
        assert_eq!(f64::from(Sign::sign_f64(-1_f64)), -1_f64);
        assert_eq!(-Sign::Negative, Sign::Positive);
        assert_eq!(-Sign::Positive, Sign::Negative);
        assert_eq!(-Sign::Zero, Sign::Zero);

        assert_eq!(i8::from(Sign::from(0_i8)), 0_i8);
        assert_eq!(i8::from(Sign::from(1_i8)), 1_i8);
        assert_eq!(i8::from(Sign::from(-3_i8)), -1_i8);

        assert_eq!(Sign::default(), Sign::Zero);

        // mul
        assert_eq!(Sign::Positive * Sign::Positive, Sign::Positive);
        assert_eq!(Sign::Negative * Sign::Positive, Sign::Negative);
        assert_eq!(Sign::Positive * Sign::Negative, Sign::Negative);
        assert_eq!(Sign::Negative * Sign::Negative, Sign::Positive);

        assert_eq!(Sign::Zero * Sign::Positive, Sign::Zero);
        assert_eq!(Sign::Zero * Sign::Negative, Sign::Zero);
        assert_eq!(Sign::Positive * Sign::Zero, Sign::Zero);
        assert_eq!(Sign::Negative * Sign::Zero, Sign::Zero);

        let mut sign = Sign::Negative;
        sign *= Sign::Negative;
        assert_eq!(sign, Sign::Positive);

        // ord
        assert_eq!(Sign::Positive.cmp(&Sign::Zero), Ordering::Greater);
        assert_eq!(Sign::Positive.cmp(&Sign::Negative), Ordering::Greater);
        assert_eq!(Sign::Negative.cmp(&Sign::Zero), Ordering::Less);
        assert_eq!(Sign::Zero.cmp(&Sign::Zero), Ordering::Equal);

        assert_eq!(
            Sign::Positive.partial_cmp(&Sign::Zero),
            Some(Ordering::Greater)
        );
        assert_eq!(
            Sign::Positive.partial_cmp(&Sign::Negative),
            Some(Ordering::Greater)
        );
        assert_eq!(
            Sign::Negative.partial_cmp(&Sign::Zero),
            Some(Ordering::Less)
        );
        assert_eq!(Sign::Zero.partial_cmp(&Sign::Zero), Some(Ordering::Equal));

        // ---
        assert_eq!(Sign::Positive.to_string(), "positive");
        assert_eq!(Sign::Negative.to_string(), "negative");
        assert_eq!(Sign::Zero.to_string(), "zero");
    }

    #[test]
    fn factorial_storage_dyn() {
        assert_eq!(FactorialStorageDyn::default(), FactorialStorageDyn::new());
    }
}
