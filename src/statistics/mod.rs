//! provide statistical tools

use std::ops::{Div, Mul, Sub};

use num_traits::Zero;
use rayon::prelude::*;

pub mod distribution;

pub use distribution::*;

/// Compute the mean from a [rayon::iter::IndexedParallelIterator].
pub fn mean_par_iter<'a, It, T>(data: It) -> T
where
    T: Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It::Item>
        + Send
        + 'a
        + Sync,
    It: IndexedParallelIterator<Item = &'a T>,
{
    mean_par_iter_val(data.cloned())
}

/// Compute the mean from a [rayon::iter::IndexedParallelIterator] by value.
pub fn mean_par_iter_val<It, T>(data: It) -> T
where
    T: Clone + Div<f64, Output = T> + std::iter::Sum<T> + std::iter::Sum<It::Item> + Send,
    It: IndexedParallelIterator<Item = T>,
{
    let len = data.len();
    let mean: T = data.sum();
    mean / len as f64
}

/// Compute the variance (squared of standard deviation) from a [rayon::iter::IndexedParallelIterator].
pub fn variance_par_iter<'a, It, T>(data: It) -> T
where
    T: Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It::Item>
        + Send
        + Sync
        + 'a
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Zero,
    It: IndexedParallelIterator<Item = &'a T> + Clone,
{
    variance_par_iter_val(data.cloned())
}

/// Compute the variance (squared of standard deviation) from a [rayon::iter::IndexedParallelIterator] by value.
pub fn variance_par_iter_val<It, T>(data: It) -> T
where
    T: Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It::Item>
        + Send
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Zero,
    It: IndexedParallelIterator<Item = T> + Clone,
{
    let [_, variance] = mean_and_variance_par_iter_val(data);
    variance
}

/// Compute the mean and variance (squared of standard deviation) from a [rayon::iter::IndexedParallelIterator].
pub fn mean_and_variance_par_iter<'a, It, T>(data: It) -> [T; 2]
where
    T: Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It::Item>
        + Send
        + Sync
        + 'a
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Zero,
    It: IndexedParallelIterator<Item = &'a T> + Clone,
{
    mean_and_variance_par_iter_val(data.cloned())
}

/// Compute the mean and variance (squared of standard deviation) from a [rayon::iter::IndexedParallelIterator] by value.
pub fn mean_and_variance_par_iter_val<It, T>(data: It) -> [T; 2]
where
    T: Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It::Item>
        + Send
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Zero,
    It: IndexedParallelIterator<Item = T> + Clone,
{
    let len = data.len();
    let (mean, mean_sqrt) = data
        .map(|el| (el.clone(), el.clone() * el))
        .reduce(|| (T::zero(), T::zero()), |a, b| (a.0 + b.0, a.1 + b.1));
    let var = (mean_sqrt - mean.clone() * mean.clone() / (len as f64)) / (len - 1) as f64;
    [mean / len as f64, var]
}

/// compute the mean the statistocal error on this value a [rayon::iter::IndexedParallelIterator].
pub fn mean_with_error_par_iter<'a, It: IndexedParallelIterator<Item = &'a f64> + Clone>(
    data: It,
) -> [f64; 2] {
    mean_with_error_par_iter_val(data.cloned())
}

/// compute the mean the statistocal error on this value a [rayon::iter::IndexedParallelIterator] by value.
pub fn mean_with_error_par_iter_val<It: IndexedParallelIterator<Item = f64> + Clone>(
    data: It,
) -> [f64; 2] {
    let len = data.len();
    let [mean, variance] = mean_and_variance_par_iter_val(data);
    [mean, (variance / len as f64).sqrt()]
}

/// compute the covariance between two [rayon::iter::IndexedParallelIterator].
/// Return `None` if the par iters are not of the same length
pub fn covariance_par_iter<'a, It1, It2, T>(data_1: It1, data_2: It2) -> Option<T>
where
    T: 'a
        + Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It1::Item>
        + Send
        + Sync
        + Mul<T, Output = T>
        + Sub<T, Output = T>,
    It1: IndexedParallelIterator<Item = &'a T> + Clone,
    It2: IndexedParallelIterator<Item = &'a T> + Clone,
    T: Zero,
{
    covariance_par_iter_val(data_1.cloned(), data_2.cloned())
}

/// compute the covariance between two [rayon::iter::IndexedParallelIterator] by value,
/// Return `None` if the par iters are not of the same length
pub fn covariance_par_iter_val<It1, It2, T>(data_1: It1, data_2: It2) -> Option<T>
where
    T: Clone
        + Div<f64, Output = T>
        + std::iter::Sum<T>
        + std::iter::Sum<It1::Item>
        + Send
        + Mul<T, Output = T>
        + Sub<T, Output = T>,
    It1: IndexedParallelIterator<Item = T> + Clone,
    It2: IndexedParallelIterator<Item = T> + Clone,
    T: Zero,
{
    if data_1.len() == data_2.len() {
        let len = data_1.len() as f64;
        let r = data_1
            .zip(data_2)
            .map(|(el_1, el_2)| (el_1.clone(), el_2.clone(), el_1 * el_2))
            .reduce(
                || (T::zero(), T::zero(), T::zero()),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            );
        Some((r.2 - r.0 * r.1 / len) / len)
    }
    else {
        None
    }
}

/// compute the mean from a collections
/// # Example
/// ```
/// use lattice_qcd_rs::statistics::mean;
/// use nalgebra::Complex;
///
/// mean(&[1_f64, 2_f64, 3_f64, 4_f64]);
/// let vec = vec![1_f64, 2_f64, 3_f64, 4_f64];
/// mean(&vec);
/// let vec_complex = vec![Complex::new(1_f64, 2_f64), Complex::new(-7_f64, -9_f64)];
/// mean(&vec_complex);
/// ```
pub fn mean<'a, T, IntoIter>(data: IntoIter) -> T
where
    T: Div<f64, Output = T> + std::iter::Sum<&'a T> + 'a,
    IntoIter: IntoIterator<Item = &'a T>,
    IntoIter::IntoIter: ExactSizeIterator,
{
    let iter = data.into_iter();
    let len = iter.len() as f64;
    let mean: T = iter.sum();
    mean / len
}

/// compute the variance (squared of standard deviation) from a collections
/// # Example
/// ```
/// use lattice_qcd_rs::statistics::variance;
/// use nalgebra::Complex;
///
/// variance(&[1_f64, 2_f64, 3_f64, 4_f64]);
/// let vec = vec![1_f64, 2_f64, 3_f64, 4_f64];
/// variance(&vec);
/// let vec_complex = vec![Complex::new(1_f64, 2_f64), Complex::new(-7_f64, -9_f64)];
/// variance(&vec_complex);
/// ```
pub fn variance<'a, T, IntoIter>(data: IntoIter) -> T
where
    T: 'a
        + Div<f64, Output = T>
        + std::iter::Sum<&'a T>
        + std::iter::Sum<T>
        + Mul<T, Output = T>
        + Clone
        + Sub<T, Output = T>,
    IntoIter: IntoIterator<Item = &'a T> + Clone,
    IntoIter::IntoIter: ExactSizeIterator,
{
    let [_, variance] = mean_and_variance(data);
    variance
}

/// compute the mean and variance (squared of standard deviation) from a collection
/// # Example
/// ```
/// use lattice_qcd_rs::statistics::mean_and_variance;
/// use nalgebra::Complex;
///
/// mean_and_variance(&[1_f64, 2_f64, 3_f64, 4_f64]);
/// let vec = vec![1_f64, 2_f64, 3_f64, 4_f64];
/// mean_and_variance(&vec);
/// let vec_complex = vec![Complex::new(1_f64, 2_f64), Complex::new(-7_f64, -9_f64)];
/// mean_and_variance(&vec_complex);
/// ```
pub fn mean_and_variance<'a, T, IntoIter>(data: IntoIter) -> [T; 2]
where
    T: 'a
        + Div<f64, Output = T>
        + std::iter::Sum<&'a T>
        + std::iter::Sum<T>
        + Mul<T, Output = T>
        + Clone
        + Sub<T, Output = T>,
    IntoIter: IntoIterator<Item = &'a T> + Clone,
    IntoIter::IntoIter: ExactSizeIterator,
{
    // often data is just a reference so cloning it is not a big deal
    let mean = mean(data.clone());
    let iter = data.into_iter();
    let len = iter.len();
    let variance = iter
        .map(|el| (el.clone() - mean.clone()) * (el.clone() - mean.clone()))
        .sum::<T>()
        / (len - 1) as f64;
    [mean, variance]
}

/// compute the mean the statistocal error on this value a slice
pub fn mean_with_error(data: &[f64]) -> [f64; 2] {
    let len = data.len();
    let [mean, variance] = mean_and_variance(data);
    [mean, (variance / len as f64).sqrt()]
}

/// compute the covariance between two slices.
/// Return `None` if the slices are not of the same length
/// # Example
/// ```
/// use lattice_qcd_rs::statistics::covariance;
/// use nalgebra::Complex;
///
/// let vec = vec![1_f64, 2_f64, 3_f64, 4_f64];
/// let array = [1_f64, 2_f64, 3_f64, 4_f64];
/// covariance(&array, &vec);
///
/// let array_complex = [Complex::new(1_f64, 2_f64), Complex::new(-7_f64, -9_f64)];
/// let vec_complex = vec![Complex::new(1_f64, 2_f64), Complex::new(-7_f64, -9_f64)];
/// covariance(&vec_complex, &array_complex);
///
/// assert!(covariance(&[], &[1_f64]).is_none());
/// ```
pub fn covariance<'a, 'b, T, IntoIter1, IntoIter2>(
    data_1: IntoIter1,
    data_2: IntoIter2,
) -> Option<T>
where
    T: 'a
        + 'b
        + Div<f64, Output = T>
        + for<'c> std::iter::Sum<&'c T>
        + std::iter::Sum<T>
        + Mul<T, Output = T>
        + Clone
        + Sub<T, Output = T>,
    IntoIter1: IntoIterator<Item = &'a T> + Clone,
    IntoIter1::IntoIter: ExactSizeIterator,
    IntoIter2: IntoIterator<Item = &'b T> + Clone,
    IntoIter2::IntoIter: ExactSizeIterator,
{
    let iter_1 = data_1.clone().into_iter();
    let iter_2 = data_2.clone().into_iter();
    if iter_1.len() == iter_2.len() {
        let len = iter_1.len();
        let mean_prod = iter_1
            .zip(iter_2)
            .map(|(el1, el2)| el1.clone() * el2.clone())
            .sum::<T>()
            / len as f64;
        Some(mean_prod - mean(data_1) * mean(data_2))
    }
    else {
        None
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    use super::*;
    #[test]
    fn mean_var() {
        let a = [1_f64; 100];
        assert_eq!(mean_and_variance_par_iter(a.par_iter()), [1_f64, 0_f64]);
        assert_eq!(mean_and_variance(&a), [1_f64, 0_f64]);
        assert_eq!(mean_par_iter(a.par_iter()), 1_f64);
        assert_eq!(variance_par_iter(a.par_iter()), 0_f64);
        assert_eq!(variance(&a), 0_f64);
        assert_eq!(mean_with_error_par_iter(a.par_iter()), [1_f64, 0_f64]);
        assert_eq!(mean_with_error(&a), [1_f64, 0_f64]);

        let a = [0_f64, 1_f64, 0_f64, 1_f64];
        assert_eq!(
            mean_and_variance_par_iter(a.par_iter()),
            [0.5_f64, 1_f64 / 3_f64]
        );
        assert_eq!(mean_and_variance(&a), [0.5_f64, 1_f64 / 3_f64]);
        assert_eq!(mean_par_iter(a.par_iter()), 0.5_f64);
        assert_eq!(variance_par_iter(a.par_iter()), 1_f64 / 3_f64);
        assert_eq!(variance(&a), 1_f64 / 3_f64);
        assert_eq!(
            mean_with_error_par_iter(a.par_iter()),
            [0.5_f64, (1_f64 / 3_f64 / 4_f64).sqrt()]
        );
        assert_eq!(
            mean_with_error(&a),
            [0.5_f64, (1_f64 / 3_f64 / 4_f64).sqrt()]
        );

        assert_eq!(covariance(&[1_f64], &[0_f64, 1_f64]), None);
        assert_eq!(
            covariance_par_iter([1_f64].par_iter(), [0_f64, 1_f64].par_iter()),
            None
        );

        let mut rng = rand::rngs::StdRng::seed_from_u64(0x45_78_93_f4_4a_b0_67_f0);
        let d = rand::distributions::Uniform::new(-1_f64, 1_f64);
        for _ in 0..100 {
            let mut vec = vec![];
            for _ in 0..100 {
                vec.push(d.sample(&mut rng));
            }
            let mut vec2 = vec![];
            for _ in 0..100 {
                vec2.push(d.sample(&mut rng));
            }
            assert!(
                (mean_and_variance(&vec)[0] - mean_and_variance_par_iter(vec.par_iter())[0]).abs()
                    < 0.00000001
            );
            assert!(
                (mean_and_variance(&vec)[1] - mean_and_variance_par_iter(vec.par_iter())[1]).abs()
                    < 0.00000001
            );
            assert!(
                (covariance(&vec, &vec2).unwrap()
                    - covariance_par_iter(vec.par_iter(), vec2.par_iter()).unwrap())
                .abs()
                    < 0.00000001
            );
        }
    }
}
