//! Contain factorial utility.
//! Also see the submodule `factorial` in [`crate::su3`]
//! for factorial used in matrix exponential function.
// TODO move into utils_lib ?

/// Internal type for factorial number.
pub type FactorialNumber = u128;

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
    /// the data is stored as a growing vector
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
    #[inline]
    pub fn build_storage(&mut self, value: usize) {
        #[allow(clippy::let_underscore_must_use)]
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

#[cfg(test)]
mod test {
    use super::{factorial, FactorialNumber, FactorialStorageDyn, MAX_NUMBER_FACTORIAL};

    /// test that the factorial pass for [`MAX_NUMBER_FACTORIAL`]
    #[allow(clippy::missing_const_for_fn)]
    #[allow(clippy::let_underscore_must_use)]
    #[test]
    fn test_factorial_pass() {
        let _: FactorialNumber = factorial(MAX_NUMBER_FACTORIAL);
    }

    #[allow(clippy::missing_const_for_fn)]
    #[allow(clippy::let_underscore_must_use)]
    #[cfg(all(feature = "overflow-test", debug_assertions))]
    #[test]
    #[should_panic(expected = "attempt to multiply with overflow")]
    /// test that the factorial overflow for `MAX_NUMBER_FACTORIAL + 1`
    fn test_factorial_bigger() {
        let _: FactorialNumber = factorial(MAX_NUMBER_FACTORIAL + 1);
    }

    #[test]
    #[should_panic(expected = "assertion failed: !overflowed")]
    /// test that the factorial overflow for `MAX_NUMBER_FACTORIAL + 1`
    fn test_factorial_overflow() {
        let n = factorial(MAX_NUMBER_FACTORIAL);
        let (_, overflowed) = n.overflowing_mul(MAX_NUMBER_FACTORIAL as u128 + 1);
        assert!(!overflowed);
    }

    #[test]
    fn factorial_storage_dyn() {
        assert_eq!(FactorialStorageDyn::default(), FactorialStorageDyn::new());
    }
}
