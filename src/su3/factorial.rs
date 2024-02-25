//! Contain factorial utilities use in matrix exponential function.
//! Also see [`crate::utils::factorial`] for more factorial utilities.
// TODO move into utils_lib ?

use std::{
    fmt::{self, Display},
    iter::FusedIterator,
};

use crate::{
    utils::{self, FactorialNumber},
    Real,
};

/// Return N such that `1/(N-7)!` < [`f64::EPSILON`].
///
/// This number is needed for the computation of exponential matrix
#[allow(clippy::as_conversions)] // no try into for f64
#[allow(clippy::cast_precision_loss)]
#[inline]
#[must_use]
pub fn factorial_size_for_exp() -> usize {
    let mut n: usize = 7;
    let mut factorial_value = 1;
    while 1_f64 / (factorial_value as f64) >= Real::EPSILON {
        n += 1;
        factorial_value *= n - 7;
    }
    n
}

/// Size of the array for [`FactorialStorageStatic`]
const FACTORIAL_STORAGE_STAT_SIZE: usize = utils::MAX_NUMBER_FACTORIAL + 1;

/// Static store for factorial number.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct FactorialStorageStatic {
    /// data store as an array (which is not growing)
    data: [FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE],
}

/// initialize the array `$data` at position `$e` as the factorial of `$e`
/// assuming `$e - 1` has been initialize or `$e == 0`.
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
    #[allow(clippy::as_conversions)] // constant function cant use try into.
    // The as is in the set_factorial_storage macro
    #[inline]
    #[must_use]
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

    /// access in O(1). Return None if `value` is bigger than 34.
    #[inline]
    #[must_use]
    pub fn try_get_factorial(&self, value: usize) -> Option<&FactorialNumber> {
        self.data.get(value)
    }

    /// Get an iterator over the factorial number form `0!` up to `34!`.
    #[inline]
    #[allow(clippy::implied_bounds_in_impls)] // no way to determine the Item of iterator otherwise
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = &FactorialNumber> + ExactSizeIterator + FusedIterator {
        self.data.iter()
    }

    /// Get the slice of factorial number.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE] {
        &self.data
    }
}

impl Default for FactorialStorageStatic {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Display for FactorialStorageStatic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "factorial Storage : ")?;
        for (i, n) in self.iter().enumerate() {
            write!(f, "{i}! = {n}")?;
            if i < self.data.len() - 1 {
                write!(f, ", ")?;
            }
        }
        Ok(())
    }
}

impl<'a> IntoIterator for &'a FactorialStorageStatic {
    type IntoIter = <&'a [u128; FACTORIAL_STORAGE_STAT_SIZE] as IntoIterator>::IntoIter;
    type Item = &'a u128;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl AsRef<[FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE]> for FactorialStorageStatic {
    #[inline]
    fn as_ref(&self) -> &[FactorialNumber; FACTORIAL_STORAGE_STAT_SIZE] {
        self.as_slice()
    }
}

/// factorial number storage in order to find the exponential in O(1) for a set storage
/// the set if for all number `N` such that `\frac{1}{(N-7)!} >= \mathrm{f64::EPSILON}`
pub const FACTORIAL_STORAGE_STAT: FactorialStorageStatic = FactorialStorageStatic::new();

#[cfg(test)]
mod test {
    use super::{
        super::N, factorial_size_for_exp, FactorialStorageStatic, FACTORIAL_STORAGE_STAT,
        FACTORIAL_STORAGE_STAT_SIZE,
    };
    use crate::{error::ImplementationError, utils::factorial};

    #[test]
    /// test that [`N`] is indeed what we need
    fn test_constant() {
        assert_eq!(N, factorial_size_for_exp() + 1);
    }

    #[test]
    fn test_factorial() -> Result<(), ImplementationError> {
        for i in 0..FACTORIAL_STORAGE_STAT_SIZE {
            assert_eq!(
                *FACTORIAL_STORAGE_STAT
                    .try_get_factorial(i)
                    .ok_or(ImplementationError::OptionWithUnexpectedNone)?,
                factorial(i)
            );
        }
        Ok(())
    }

    #[test]
    fn factorial_storage() {
        let storage = FactorialStorageStatic::new();
        for (i, f) in storage.data.iter().enumerate() {
            assert_eq!(factorial(i), *f);
        }
        assert!(storage.try_get_factorial(35).is_none());
        assert_eq!(storage.try_get_factorial(34), Some(&factorial(34)));
    }
}
