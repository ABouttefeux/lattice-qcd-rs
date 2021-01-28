
//! Utils function and structure

type FactorialNumber = u128;

/// Smallest number sur that (n+1)! overflow u128.
pub const MAX_NUMBER_FACTORIAL: usize = 34;

/// return n!.
///
/// #Example
/// ```
/// # use lattice_qcd_rs::utils::factorial;
/// assert_eq!(factorial(4), 24);
/// assert_eq!(factorial(6), 720);
/// ```
pub const fn factorial(n: usize) -> FactorialNumber {
    if n == 0 {
        return 1;
    }
    else{
        return n as FactorialNumber * factorial(n - 1);
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
            self.data.push(self.data[i - 1] * i as FactorialNumber);
        }
        return self.data[value];
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
            value_m *= (i + 1) as FactorialNumber;
        }
        return value_m;
    }
}

#[cfg(test)]
#[test]
/// test that the factorial pass for MAX_NUMBER_FACTORIAL
fn test_factorial_pass() {
    factorial(MAX_NUMBER_FACTORIAL);
}

#[cfg(test)]
#[test]
#[should_panic]
/// test that the factorial overflow for MAX_NUMBER_FACTORIAL + 1
fn test_factorial_bigger() {
    factorial(MAX_NUMBER_FACTORIAL + 1);
}
