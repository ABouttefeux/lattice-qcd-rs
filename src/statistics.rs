
//! provide statistical tools

use std::ops::{Sub, Div, Mul};
use rayon::prelude::*;
use num_traits::Zero;

/// Compute the mean from a [rayon::iter::IndexedParallelIterator].
pub fn mean_par_iter<'a ,It , T>(data: It) -> T
    where T: Clone + Div<f64, Output = T> + std::iter::Sum<T> +std::iter::Sum<It::Item> + Send +'a ,
    It: IndexedParallelIterator<Item = &'a T>,
{
    let len = data.len();
    let mean: T = data.sum();
    mean / len as f64
}

/// Compute the mean from a [rayon::iter::IndexedParallelIterator].
pub fn mean_par_iter_val <It , T>(data: It) -> T
    where T: Clone + Div<f64, Output = T> + std::iter::Sum<T> +std::iter::Sum<It::Item> + Send,
    It: IndexedParallelIterator<Item = T>,
{
    let len = data.len();
    let mean: T = data.sum();
    mean / len as f64
}



/// Compute the variance (squared of standard deviation) from a [rayon::iter::IndexedParallelIterator].
pub fn variance_par_iter<'a, It,  T>(data: It) -> T
    where T: Clone + Div<f64, Output = T> + std::iter::Sum<T> +std::iter::Sum<It::Item> + Send + Sync + 'a + Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Zero,
    It: IndexedParallelIterator<Item = &'a T> + Clone,
{
    let [_, variance] = mean_and_variance_par_iter(data);
    variance
}

/// Compute the mean and variance (squared of standard deviation) from a [rayon::iter::IndexedParallelIterator].
pub fn mean_and_variance_par_iter<'a, It, T>(data: It) -> [T; 2]
    where T: Clone + Div<f64, Output = T> + std::iter::Sum<T> +std::iter::Sum<It::Item> + Send + Sync + 'a + Sub<T, Output = T>,
    T: Mul<T, Output = T>,
    It: IndexedParallelIterator<Item = &'a T> + Clone,
    T: Zero,
{
    let len = data.len();
    let (mean, mean_sqrt) = data.map(|el| (el.clone(), el.clone() * el.clone()) )
        .reduce( || (T::zero(), T::zero()) , |a, b| (a.0 + b.0, a.1 + b.1));
    let var = (mean_sqrt - mean.clone() * mean.clone() / (len as f64)) / (len - 1 ) as f64;
    [mean/ len as f64, var]
}

/// compute the mean the statistocal error on this value a [rayon::iter::IndexedParallelIterator]
pub fn mean_with_error_par_iter<'a, It: IndexedParallelIterator<Item = &'a f64> + Clone>(data: It) -> [f64; 2]
{
    let len = data.len();
    let [mean, variance] = mean_and_variance_par_iter(data);
    [mean, (variance / len as f64).sqrt()]
}


/// compute the auto correlaction from a [rayon::iter::IndexedParallelIterator]
pub fn auto_correlation_par_iter<'a, It1, It2, T>(data_1: It1, data_2: It2) -> Option<T>
    where T: Clone + Div<f64, Output = T> + std::iter::Sum<T> +std::iter::Sum<It1::Item> + Send + Sync + Mul<T, Output = T> + 'a + Sub<T, Output = T>,
    It1: IndexedParallelIterator<Item = &'a T> + Clone,
    It2: IndexedParallelIterator<Item = &'a T> + Clone,
    T: Zero,
{
    if data_1.len() == data_2.len() {
        let len = data_1.len() as f64;
        let r = data_1.zip(data_2)
            .map(|(el_1, el_2)| (el_1.clone(), el_2.clone(), el_1.clone() * el_2.clone()))
            .reduce(|| (T::zero(), T::zero(), T::zero()) , |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        Some( (r.2 - r.0 * r.1 / len) / len)
    }
    else {
        None
    }
    
}

/// compute the mean from a slice
pub fn mean<'a, T>(data: &'a [T]) -> T
    where T: Div<f64, Output = T> + std::iter::Sum<&'a T>,
{
    let mean: T = data.iter().sum();
    mean / data.len() as f64
}

/// compute the variance (squared of standard deviation) from a slice
pub fn variance<'a, T>(data: &'a [T]) -> T
    where T: Div<f64, Output = T> +std::iter::Sum<&'a T> + std::iter::Sum<T> + Mul<T, Output = T> + Clone + Sub<T, Output = T>,
{
    let [_, variance] = mean_and_variance(data);
    variance
}

/// compute the mean and variance (squared of standard deviation) from a slice
pub fn mean_and_variance<'a, T>(data: &'a [T]) -> [T; 2]
    where T: Div<f64, Output = T> + std::iter::Sum<&'a T> + std::iter::Sum<T> + Mul<T, Output = T> + Clone + Sub<T, Output = T>,
{
    let mean = mean(data);
    let variance = data.iter().map(|el| (el.clone() - mean.clone()) * (el.clone() - mean.clone()) ).sum::<T>() / (data.len() - 1) as f64;
    [mean, variance]
}

/// compute the mean the statistocal error on this value a slice
pub fn mean_with_error<It: IndexedParallelIterator<Item = f64> + Clone>(data: &[f64]) -> [f64; 2]
{
    let len = data.len();
    let [mean, variance] = mean_and_variance(data);
    [mean, (variance / len as f64).sqrt()]
}

/// compute the auto correlaction from a slice
pub fn auto_correlation<'a, T>(data_1: &'a [T],data_2: &'a [T]) -> Option<T>
    where T: Div<f64, Output = T> + std::iter::Sum<&'a T> + std::iter::Sum<T> + Mul<T, Output = T> + Clone + Sub<T, Output = T>,
{
    if data_1.len() == data_2.len() {
        let mean_prod = data_1.iter().zip(data_2.iter()).map(|(el1, el2)| el1.clone() * el2.clone() ).sum::<T>() / data_2.len() as f64;
        Some(mean_prod - mean(data_1) * mean(data_2))
    }
    else {
        None
    }
    
}

#[cfg(test)]
mod test {
    use super::*;
    use rand_distr::Distribution;
    #[test]
    fn mean_var() {
        let a = [1_f64; 100];
        assert_eq!(mean_and_variance_par_iter(a.par_iter()), [1_f64, 0_f64]);
        assert_eq!(mean_and_variance(&a), [1_f64, 0_f64]);
        let a = [0_f64, 1_f64, 0_f64, 1_f64];
        assert_eq!(mean_and_variance_par_iter(a.par_iter()), [0.5_f64, 1_f64 / 3_f64]);
        assert_eq!(mean_and_variance(&a), [0.5_f64, 1_f64 / 3_f64]);
        let mut rng = rand::thread_rng();
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
            assert!( (mean_and_variance(&vec)[0] - mean_and_variance_par_iter(vec.par_iter())[0]).abs() < 0.00000001 );
            assert!( (mean_and_variance(&vec)[1] - mean_and_variance_par_iter(vec.par_iter())[1]).abs() < 0.00000001 );
            assert!((auto_correlation(&vec, &vec2).unwrap() - auto_correlation_par_iter(vec.par_iter(), vec2.par_iter()).unwrap()).abs() < 0.00000001);
        }
    }
}
