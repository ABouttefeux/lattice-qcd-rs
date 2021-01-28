//! Work in progress
//!
//! module to represent numbers.

/*
pub struct Rng<T, V>
    where T: rand_distr::Distribution<V>,
{
    rng: rand::rngs::ThreadRng,
    distribution: T
}

impl<T, V> Rng<T,V>
    where T: rand_distr::Distribution<V>,
{
    pub fn get_random_number(&mut self) -> V {
        self.sample(&mut self.rng)
    }
    
    pub fn new(rng: rand::rngs::ThreadRng, distribution: T) -> Self{
        Self {rng, distribution}
    }
}
*/

//! Fix point number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPointNumber<I, D>
    where I: num_traits::sign::Signed + std::cmp::Ord + Copy,
    D : num_traits::sign::Unsigned + std::cmp::Ord + Copy,
{
    integer : I,
    decimal : D,
}


impl<I, D> FixedPointNumber<I, D>
    where I: num_traits::sign::Signed + std::cmp::Ord + Copy,
    D : num_traits::sign::Unsigned + std::cmp::Ord + Copy,
{
    pub fn integer(&self) -> I {
        self.integer
    }
    
    pub fn decimal(&self) -> D {
        self.decimal
    }
}

pub type I32U128 = FixedPointNumber<i32, u128>;

/*
impl<I, D> std::ops::Neg for FixedPointNumber<I, D>
    where I: num_traits::sign::Signed + std::cmp::Ord + Copy,
    D : num_traits::sign::Unsigned + std::cmp::Ord + Copy,
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
    where I: num_traits::sign::Signed + std::cmp::Ord + Copy,
    D : num_traits::sign::Unsigned + std::cmp::Ord + Copy,
{
    
}

impl<I, D> num_traits::real::Real for FixedPointNumber<I, D>
    where I: num_traits::sign::Signed + std::cmp::Ord + Copy,
    D : num_traits::sign::Unsigned + std::cmp::Ord + Copy,
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
