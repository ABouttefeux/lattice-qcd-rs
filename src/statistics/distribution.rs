//! Some statistical distribution used by other part of the library

use std::ops::{Add, Div, Mul, Neg, Sub};

use num_traits::{Float, FloatConst, One, Zero};
use rand::distributions::Uniform;
use rand_distr::Distribution;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::super::{su2, su3, CMatrix2, CMatrix3, Real};

/// Distribution given by `x^2 e^{- 2 a x^2}`, `x >= 0` where `x` is the random variable and `a` a parameter of the distribution.
///
/// # Example
/// ```
/// use lattice_qcd_rs::error::ImplementationError;
/// use lattice_qcd_rs::statistics::ModifiedNormal;
/// use rand::{Rng, SeedableRng};
///
/// # fn main() -> Result<(), ImplementationError> {
/// let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// let mn = ModifiedNormal::new(0.5_f64).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let r_number = rng.sample(&mn);
/// #
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Copy, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ModifiedNormal<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
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
        + FloatConst
        + Zero
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
{
    getter_copy!(
        /// Returns the parameter `a`.
        pub const,
        param_exp,
        T
    );

    /// Create the distribution. `param_exp` should be strictly greater than 0 an be finite and a number.
    /// Otherwise return [`None`].
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

impl<T> std::fmt::Display for ModifiedNormal<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Float
        + FloatConst
        + PartialOrd
        + std::fmt::Display,
    rand::distributions::OpenClosed01: Distribution<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "modified normal distribution with parameter {}",
            self.param_exp()
        )
    }
}

/// Distribution for the Heat Bath methods with the parameter `param_exp = beta * sqrt(det(A))`.
///
/// With distribution `dP(X) = 1/(2 \pi^2) d \cos(\theta) d\phi dx_0 \sqrt(1-x_0^2) e^{param_exp x_0}`.
///
/// # Example
/// ```
/// use lattice_qcd_rs::error::ImplementationError;
/// use lattice_qcd_rs::statistics::HeatBathDistribution;
/// use nalgebra::{Complex, Matrix2};
/// use rand::{Rng, SeedableRng};
///
/// # fn main() -> Result<(), ImplementationError> {
/// let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// let heat_bath =
///     HeatBathDistribution::new(0.5_f64).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let r_matrix: Matrix2<Complex<f64>> = rng.sample(&heat_bath);
/// #
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Copy, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HeatBathDistribution<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Neg<Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
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
        + Sub<T, Output = T>
        + Neg<Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    getter_copy!(
        /// Returns the parameter `param_exp`.
        pub const,
        param_exp,
        T
    );

    /// Create the distribution. `param_exp` should be strictly greater than 0 an be finite and a number.
    /// Otherwise return [`None`].
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
        // TODO make a function to reduce copy of code with su2::get_random_su2_close_to_unity

        let distr_norm = HeatBathDistributionNorm::new(self.param_exp()).expect("unreachable");
        // unreachable because self.param_exp() > 0 which Create the distribution
        let x0: f64 = rng.sample(&distr_norm);
        let uniform = Uniform::new(-1_f64, 1_f64);
        let mut x_unorm = na::Vector3::from_fn(|_, _| rng.sample(&uniform));
        while x_unorm.norm() <= f64::EPSILON {
            x_unorm = na::Vector3::from_fn(|_, _| rng.sample(&uniform));
        }
        let x =
            x_unorm.try_normalize(f64::EPSILON).expect("unreachable") * (1_f64 - x0 * x0).sqrt();
        // unreachable because the while loop above guarantee that the norm is bigger than [`f64::EPSILON`]
        su2::complex_matrix_from_vec(x0, x)
    }
}

impl<T> std::fmt::Display for HeatBathDistribution<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Neg<Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + PartialOrd
        + std::fmt::Display,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "heat bath distribution with parameter {}",
            self.param_exp()
        )
    }
}

/// Distribution for the norm of the SU2 adjoint to generate the [`HeatBathDistribution`] with the parameter
/// `param_exp = beta * sqrt(det(A))`.
///
/// With distribution `dP(x) = dx \sqrt(1-x_0^2) e^{-2 param_exp x^2}`
/// # Example
/// ```
/// use lattice_qcd_rs::error::ImplementationError;
/// use lattice_qcd_rs::statistics::HeatBathDistributionNorm;
/// use rand::{Rng, SeedableRng};
///
/// # fn main() -> Result<(), ImplementationError> {
/// let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// let heat_bath = HeatBathDistributionNorm::new(0.5_f64)
///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let r_number = rng.sample(&heat_bath);
/// #
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Copy, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HeatBathDistributionNorm<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Neg<Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
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
        + Sub<T, Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + PartialOrd,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    getter_copy!(
        /// return the parameter `param_exp`.
        pub const,
        param_exp,
        T
    );

    /// Create the distribution. `param_exp` should be strictly greater than 0 an be finite and a number.
    /// Otherwise return [`None`].
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
        + Sub<T, Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
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

impl<T> std::fmt::Display for HeatBathDistributionNorm<T>
where
    T: One
        + Div<T, Output = T>
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Neg<Output = T>
        + Sub<T, Output = T>
        + Float
        + FloatConst
        + Zero
        + rand_distr::uniform::SampleUniform
        + PartialOrd
        + std::fmt::Display,
    rand::distributions::OpenClosed01: Distribution<T>,
    Uniform<T>: Distribution<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "heat bath norm distribution with parameter {}",
            self.param_exp()
        )
    }
}

/// used to generates matrix close the unit, for su2 close to +/-1, see [`su2::random_su2_close_to_unity`]
/// and for su(3) `[su3::get_r] (+/- 1) * [su3::get_s] (+/- 1) * [su3::get_t] (+/- 1)`
///
/// # Example
/// ```
/// use lattice_qcd_rs::error::ImplementationError;
/// use lattice_qcd_rs::statistics::CloseToUnit;
/// use nalgebra::{Complex, Matrix2, Matrix3};
/// use rand::{Rng, SeedableRng};
///
/// # fn main() -> Result<(), ImplementationError> {
/// let mut rng = rand::rngs::StdRng::seed_from_u64(0);
/// let close_to_unit =
///     CloseToUnit::new(0.5_f64).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let r_matrix: Matrix2<Complex<f64>> = rng.sample(&close_to_unit);
/// let r_matrix: Matrix3<Complex<f64>> = rng.sample(&close_to_unit);
/// #
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CloseToUnit {
    spread_parameter: Real,
}

impl CloseToUnit {
    getter_copy!(
        /// Get the spread parameter
        pub const,
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
        su2::random_su2_close_to_unity(self.spread_parameter, rng)
    }
}

impl Distribution<CMatrix3> for CloseToUnit {
    fn sample<R>(&self, rng: &mut R) -> CMatrix3
    where
        R: rand::Rng + ?Sized,
    {
        su3::random_su3_close_to_unity(self.spread_parameter, rng)
    }
}

impl std::fmt::Display for CloseToUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "distribution closed to unit with spread parameter {}",
            self.spread_parameter()
        )
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    use super::*;

    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    fn modified_normal() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);

        for param in &[0.1_f64, 0.5_f64, 1_f64, 10_f64] {
            let mn = ModifiedNormal::new(*param).unwrap();

            for _ in 0_u32..1000_u32 {
                assert!(rng.sample(&mn) >= 0_f64);
            }
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn distribution_creation() {
        assert!(ModifiedNormal::new(0_f64).is_none());
        assert!(ModifiedNormal::new(-1_f64).is_none());
        assert!(ModifiedNormal::new(f64::NAN).is_none());
        assert!(ModifiedNormal::new(f64::INFINITY).is_none());
        assert!(ModifiedNormal::new(0.1_f64).is_some());
        assert!(ModifiedNormal::new(2_f32).is_some());

        let param = 0.5_f64;
        let mn = ModifiedNormal::new(param).unwrap();
        assert_eq!(mn.param_exp(), param);
        assert_eq!(
            mn.to_string(),
            "modified normal distribution with parameter 0.5"
        );

        assert!(HeatBathDistributionNorm::new(0_f64).is_none());
        assert!(HeatBathDistributionNorm::new(-1_f64).is_none());
        assert!(HeatBathDistributionNorm::new(f64::NAN).is_none());
        assert!(HeatBathDistributionNorm::new(f64::INFINITY).is_none());
        assert!(HeatBathDistributionNorm::new(0.1_f64).is_some());
        assert!(HeatBathDistributionNorm::new(2_f32).is_some());

        let heat_bath_norm = HeatBathDistributionNorm::new(param).unwrap();
        assert_eq!(heat_bath_norm.param_exp(), param);
        assert_eq!(
            heat_bath_norm.to_string(),
            "heat bath norm distribution with parameter 0.5"
        );

        assert!(HeatBathDistribution::new(0_f64).is_none());
        assert!(HeatBathDistribution::new(-1_f64).is_none());
        assert!(HeatBathDistribution::new(f64::NAN).is_none());
        assert!(HeatBathDistribution::new(f64::INFINITY).is_none());
        assert!(HeatBathDistribution::new(0.1_f64).is_some());
        assert!(HeatBathDistribution::new(2_f32).is_some());

        let heat_bath = HeatBathDistribution::new(param).unwrap();
        assert_eq!(heat_bath.param_exp(), param);
        assert_eq!(
            heat_bath.to_string(),
            "heat bath distribution with parameter 0.5"
        );

        assert!(CloseToUnit::new(0_f64).is_none());
        assert!(CloseToUnit::new(-1_f64).is_none());
        assert!(CloseToUnit::new(f64::NAN).is_none());
        assert!(CloseToUnit::new(f64::INFINITY).is_none());
        assert!(CloseToUnit::new(2_f64).is_none());
        assert!(CloseToUnit::new(0.5_f64).is_some());

        let cu = CloseToUnit::new(param).unwrap();
        assert_eq!(cu.spread_parameter(), param);
        assert_eq!(
            cu.to_string(),
            "distribution closed to unit with spread parameter 0.5"
        );
    }
}
