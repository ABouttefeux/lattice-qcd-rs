//! Work in progress
//!
//! module to represent numbers.

use num_traits::{Signed, Unsigned};

/// Fix point number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedPointNumber<I, D>
where
    I: Signed + Ord + Copy,
    D: Unsigned + Ord + Copy,
{
    integer: I,
    decimal: D,
}

impl<I, D> FixedPointNumber<I, D>
where
    I: Signed + Ord + Copy,
    D: Unsigned + Ord + Copy,
{
    /// Get tje integer part of the number
    #[inline]
    #[must_use]
    pub const fn integer(&self) -> I {
        self.integer
    }

    /// Get the decimal part of the number as it is stored (as raw data)
    #[inline]
    #[must_use]
    pub const fn decimal(&self) -> D {
        self.decimal
    }
}

/// Fixe point number represented by i32 for the integer part and 128 bits (16 bytes) as the decimal part
pub type I32U128 = FixedPointNumber<i32, u128>;

/*
impl<I, D> std::ops::Neg for FixedPointNumber<I, D>
    where I: Signed + Ord + Copy,
    D : Unsigned + Ord + Copy,
{
    type Output = Self;

     fn neg(mut self) -> Self::Output {
        self.integer = - self.integer;
        if self.decimal != 0 {
            self.integer += 1;
            self.decimal = D::MAX - self.decimal + 1;
        }
        return self;
     }
}
*/

/*
impl<I, D> num_traits::Num for FixedPointNumber<I, D>
    where I: Signed + Ord + Copy,
    D : Unsigned + Ord + Copy,
{

}

impl<I, D> num_traits::real::Real for FixedPointNumber<I, D>
    where I: Signed + Ord + Copy,
    D : Unsigned + Ord + Copy,
{
    fn min_value() -> Self {
        Self {
            integer: I::MIN,
            decimal: D::MIN,
        }
    }

    fn max_value() -> Self {
        Self {
            integer: I::MAX,
            decimal: D::MAX,
        }
    }

    fn epsilon() -> Self{
        Self {
            integer: 0,
            decimal: 1,
        }
    }

    fn min_positive_value() -> Self{
        Self {
            integer: 0,
            decimal: 1,
        }
    }

    fn floor(self) -> Self{
        Self {
            integer: self.integer,
            decimal: 0,
        }
    }

    fn ceil(self) -> Self {
        let mut integer = self.integer;
        if self.decimal > 0 {
            integer += 1
        }
        Self {
            integer,
            decimal: 0,
        }
    }

    fn round(self) -> Self {
        let mut integer = self.integer;
        if self.decimal >= D::MAX / 2 {
            integer += 1
        }
        Self {
            integer,
            decimal: 0,
        }
    }
}

*/
