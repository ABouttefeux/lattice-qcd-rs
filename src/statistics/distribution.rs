//! Some statistical distribution used by other part of the library

use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{Float, FloatConst, One, Zero};
use rand::distributions::Uniform;
use rand_distr::Distribution;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::super::{su2, su3, CMatrix2, CMatrix3, Real};

/// Distribution given by `x^2 e^{- 2 a x^2}`, `x >= 0` where `x` is the random variable and `a` a parameter of the distribution
#[derive(Clone, Debug, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ModifiedNormal<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
{
    param_exp: T,
}

impl<T> ModifiedNormal<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + Zero
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
{
    getter_copy!(
        /// return the parameter `a`.
        param_exp,
        T
    );

    /// Create the distribution. `param_exp` should be strictly greater than 0 an be finite and a number.
    /// Otherwise return None.
    pub fn new(param_exp: T) -> Option<Self> {
        if param_exp.le(&T::zero()) || param_exp.is_infinite() || param_exp.is_nan() {
            return None;
        }
        Some(Self { param_exp })
    }
}

impl<T> Distribution<T> for ModifiedNormal<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + rand_distr::uniform::SampleUniform
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
{
    fn sample<R>(&self, rng: &mut R) -> T
    where
        R: rand::Rng + ?Sized,
    {
        let mut r = [T::one(); 3];
        for element in r.iter_mut() {
            *element = rng.sample(rand::distributions::OpenClosed01);
        }
        let two = T::one() + T::one();
        (-(r[0].ln() + (two * T::PI() * r[1]).cos().powi(2) * r[2].ln()) / (two * self.param_exp()))
            .sqrt()
    }
}

/// Distribution for the Heat Bath methods with the parameter `param_exp = beta * sqrt(det(A))`.
///
/// With distribution `dP(X) = 1/(2 \pi^2) d \cos(\theta) d\phi dx_0 \sqrt(1-x_0^2) e^{param_exp x_0}`
#[derive(Clone, Debug, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HeatBathDistribution<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + Sub<T, Output = T>
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    param_exp: T,
}

impl<T> HeatBathDistribution<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + Sub<T, Output = T>
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    getter_copy!(
        /// return the parameter `param_exp`.
        param_exp,
        T
    );

    /// Create the distribution. `param_exp` should be strictly greater than 0 an be finite and a number.
    /// Otherwise return None.
    pub fn new(param_exp: T) -> Option<Self> {
        if param_exp.le(&T::zero()) || param_exp.is_infinite() || param_exp.is_nan() {
            return None;
        }
        Some(Self { param_exp })
    }
}

impl Distribution<CMatrix2> for HeatBathDistribution<f64> {
    fn sample<R>(&self, rng: &mut R) -> CMatrix2
    where
        R: rand::Rng + ?Sized,
    {
        // TODO make a functrion to reduce copy of code with su2::get_random_su2_close_to_unity

        let distr_norm = HeatBathDistributionNorm::new(self.param_exp()).expect("unreacheable");
        // unreacheable because self.param_exp() > 0 which Create the distribution
        let x0: f64 = rng.sample(&distr_norm);
        let uniform = Uniform::new(-1_f64, 1_f64);
        let mut x_unorm = na::Vector3::from_fn(|_, _| rng.sample(&uniform));
        while x_unorm.norm() <= f64::EPSILON {
            x_unorm = na::Vector3::from_fn(|_, _| rng.sample(&uniform));
        }
        let x =
            x_unorm.try_normalize(f64::EPSILON).expect("unreacheable") * (1_f64 - x0 * x0).sqrt();
        // unreacheable because the while loop above guarentee that the norm is bigger than [`f64::EPSILON`]
        su2::get_complex_matrix_from_vec(x0, x)
    }
}

/// Distribution for the norm of the SU2 adjoint to generate the [`HeatBathDistribution`] with the parameter `param_exp = beta * qrt(det(A))`.
///
/// With distribution `dP(x) = dx \sqrt(1-x_0^2) e^{-2 param_exp x^2}`
#[derive(Clone, Debug, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HeatBathDistributionNorm<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + Sub<T, Output = T>
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    param_exp: T,
}

impl<T> HeatBathDistributionNorm<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + Sub<T, Output = T>
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    getter_copy!(
        /// return the parameter `param_exp`.
        param_exp,
        T
    );

    /// Create the distribution. `param_exp` should be strictly greater than 0 an be finite and a number.
    /// Otherwise return None.
    pub fn new(param_exp: T) -> Option<Self> {
        if param_exp.le(&T::zero()) || param_exp.is_infinite() || param_exp.is_nan() {
            return None;
        }
        Some(Self { param_exp })
    }
}

impl<T> Distribution<T> for HeatBathDistributionNorm<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + Copy
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + Sub<T, Output = T>
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    fn sample<R>(&self, rng: &mut R) -> T
    where
        R: rand::Rng + ?Sized,
    {
        let two = T::one() + T::one();
        // the unwrap is OK because we verify that the param is not zero at creation.
        let distributions = ModifiedNormal::new(self.param_exp()).unwrap();
        loop {
            let r = rng.sample(Uniform::new(T::zero(), T::one()));
            let lambda = rng.sample(distributions);
            if r.powi(2) <= T::one() - lambda.powi(2) {
                return T::one() - two * lambda.powi(2);
            }
        }
    }
}

/// used to generates matrix close the unit, for su2 close to +/-1, see [`su2::get_random_su2_close_to_unity`] and for su(3) `[su3::get_r] (+/- 1) * [su3::get_s] (+/- 1) * [su3::get_t] (+/- 1)`
#[derive(Clone, Debug, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CloseToUnit {
    spread_parameter: Real,
}

impl CloseToUnit {
    getter_copy!(
        const,
        /// Get the spread parameter
        spread_parameter,
        f64
    );

    /// Create a new distribution, spread_parameter should be in `(0,1)` 0 and 1 excluded
    pub fn new(spread_parameter: Real) -> Option<Self> {
        if spread_parameter <= 0_f64 || spread_parameter >= 1_f64 || spread_parameter.is_nan() {
            return None;
        }
        Some(Self { spread_parameter })
    }
}

impl Distribution<CMatrix2> for CloseToUnit {
    fn sample<R>(&self, rng: &mut R) -> CMatrix2
    where
        R: rand::Rng + ?Sized,
    {
        su2::get_random_su2_close_to_unity(self.spread_parameter, rng)
    }
}

impl Distribution<CMatrix3> for CloseToUnit {
    fn sample<R>(&self, rng: &mut R) -> CMatrix3
    where
        R: rand::Rng + ?Sized,
    {
        su3::get_random_su3_close_to_unity(self.spread_parameter, rng)
    }
}
