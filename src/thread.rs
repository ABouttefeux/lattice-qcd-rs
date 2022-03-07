//! tool for easy use of mutli threading.

use std::{
    any::Any,
    collections::HashMap,
    error::Error,
    fmt::{self, Display, Formatter},
    hash::Hash,
    iter::Iterator,
    sync::{mpsc, Arc, Mutex},
    vec::Vec,
};

use crossbeam::thread;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::lattice::{LatticeCyclique, LatticeElementToIndex};

/// Multithreading error.
///
/// This can be converted to [`ThreadError`] which is more convenient to use keeping only the case
/// with [`String`] and [`&str`] messages.
#[derive(Debug)]
#[non_exhaustive]
pub enum ThreadAnyError {
    /// Tried to run some jobs with 0 threads
    ThreadNumberIncorect,
    /// One or more of the threads panicked. Inside the [`Box`] is the panic message.
    /// see [`run_pool_parallel`] example.
    Panic(Vec<Box<dyn Any + Send + 'static>>),
}

impl Display for ThreadAnyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadNumberIncorect => write!(f, "number of thread is incorrect"),
            Self::Panic(any) => {
                let n = any.len();
                if n == 0 {
                    write!(f, "0 thread panicked")?;
                }
                else if n == 1 {
                    write!(f, "a thread panicked with")?;
                }
                else {
                    write!(f, "{} threads panicked with [", n)?;
                }

                for (index, element_any) in any.iter().enumerate() {
                    if let Some(string) = element_any.downcast_ref::<String>() {
                        write!(f, "\"{}\"", string)?;
                    }
                    else if let Some(string) = element_any.downcast_ref::<&str>() {
                        write!(f, "\"{}\"", string)?;
                    }
                    else {
                        write!(f, "{:?}", element_any)?;
                    }

                    if index < any.len() - 1 {
                        write!(f, " ,")?;
                    }
                    else if n > 1 {
                        write!(f, "]")?;
                    }
                }

                Ok(())
            }
        }
    }
}

impl Error for ThreadAnyError {}

/// Multithreading error with a string panic message.
///
/// It is more convenient to use compared to [`ThreadAnyError`] and can be converted from it.
/// It convert message of type [`String`] and [`&str`] otherwise set it to None.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ThreadError {
    /// Tried to run some jobs with 0 threads
    ThreadNumberIncorect,
    /// One of the thread panicked with the given messages.
    /// see [`run_pool_parallel`] example.
    Panic(Vec<Option<String>>),
}

impl Display for ThreadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadNumberIncorect => write!(f, "number of thread is incorrect"),
            Self::Panic(strings) => {
                let n = strings.len();
                if n == 0 {
                    // this should not be used but it is possible to create an instance with an empty vec.
                    write!(f, "0 thread panicked")?;
                }
                else if n == 1 {
                    write!(f, "a thread panicked with")?;
                }
                else {
                    write!(f, "{} threads panicked with [", n)?;
                }

                for (index, string) in strings.iter().enumerate() {
                    if let Some(string) = string {
                        write!(f, "\"{}\"", string)?;
                    }
                    else {
                        write!(f, "None")?;
                    }

                    if index < strings.len() - 1 {
                        write!(f, " ,")?;
                    }
                    else if n > 1 {
                        write!(f, "]")?;
                    }
                }

                Ok(())
            }
        }
    }
}

impl Error for ThreadError {}

impl From<ThreadAnyError> for ThreadError {
    #[allow(clippy::manual_map)] // clarity / false positive ?
    fn from(f: ThreadAnyError) -> Self {
        match f {
            ThreadAnyError::ThreadNumberIncorect => Self::ThreadNumberIncorect,
            ThreadAnyError::Panic(any) => Self::Panic(
                any.iter()
                    .map(|element| {
                        if let Some(string) = element.downcast_ref::<String>() {
                            Some(string.clone())
                        }
                        else if let Some(string) = element.downcast_ref::<&str>() {
                            Some(string.to_string())
                        }
                        else {
                            None
                        }
                    })
                    .collect(),
            ),
        }
    }
}

impl From<ThreadError> for ThreadAnyError {
    fn from(f: ThreadError) -> Self {
        match f {
            ThreadError::ThreadNumberIncorect => Self::ThreadNumberIncorect,
            ThreadError::Panic(strings) => Self::Panic(
                strings
                    .iter()
                    .map(|string| -> Box<dyn Any + Send + 'static> {
                        if let Some(string) = string {
                            Box::new(string.clone())
                        }
                        else {
                            Box::new("".to_string())
                        }
                    })
                    .collect(),
            ),
        }
    }
}

/// run jobs in parallel.
///
/// The pool of job is given by `iter`. the job is given by `closure` that have the form `|key,common_data| -> Data`.
/// `number_of_thread` determine the number of job done in parallel and should be greater than 0,
/// otherwise return [`ThreadAnyError::ThreadNumberIncorect`].
/// `capacity` is used to determine the capacity of the [`HashMap`] upon initisation (see [`HashMap::with_capacity`])
///
/// # Errors
/// Returns [`ThreadAnyError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadAnyError::Panic`] if a thread panicked. Containt the panic message.
///
/// # Example
/// let us computes the value of `i^2 * c` for i in \[2,9999\] with 4 threads
/// ```
/// # use lattice_qcd_rs::thread::run_pool_parallel;
/// # use std::error::Error;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let iter = 2..10000;
/// let c = 5;
/// // we could have put 4 inside the closure but this demonstrate how to use common data
/// let result = run_pool_parallel(iter, &c, &|i, c| i * i * c, 4, 10000 - 2)?;
/// assert_eq!(*result.get(&40).unwrap(), 40 * 40 * c);
/// assert_eq!(result.get(&1), None);
/// # Ok(())
/// # }
/// ```
/// In the next example a thread will panic, we demonstrate the return type.
/// ```should_panic
/// # use lattice_qcd_rs::thread::{run_pool_parallel, ThreadAnyError};
/// let iter = 0..10;
/// let result = run_pool_parallel(iter, &(), &|_, _| panic!("{}", "panic message"), 4, 10);
/// match result {
///     Ok(_) => {}
///     Err(err) => panic!("{}", err),
/// }
/// ```
/// This give the following panic message
/// ```textrust
/// stderr:
/// thread '<unnamed>' panicked at 'panic message', src\thread.rs:6:51
/// note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
/// thread '<unnamed>' panicked at 'panic message', src\thread.rs:6:51
/// thread '<unnamed>' panicked at 'panic message', src\thread.rs:6:51
/// thread '<unnamed>' panicked at 'panic message', src\thread.rs:6:51
/// thread 'main' panicked at '4 threads panicked with ["panic message" ,"panic message" ,"panic message" ,"panic message"]', src\thread.rs:9:17
/// ```
pub fn run_pool_parallel<Key, Data, CommonData, F>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    number_of_thread: usize,
    capacity: usize,
) -> Result<HashMap<Key, Data>, ThreadAnyError>
where
    CommonData: Sync,
    Key: Eq + Hash + Send + Clone + Sync,
    Data: Send,
    F: Sync + Clone + Fn(&Key, &CommonData) -> Data,
{
    run_pool_parallel_with_initialisation_mutable(
        iter,
        common_data,
        &|_, key, common| closure(key, common),
        &|| (),
        number_of_thread,
        capacity,
    )
}

/// run jobs in parallel. Similar to [`run_pool_parallel`] but with initiation.
///
/// see [`run_pool_parallel`]. Moreover let some data to be initialize per thread.
/// closure_init is run once per thread and store inside a mutable data which closure can modify.
///
/// # Errors
/// Returns [`ThreadAnyError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadAnyError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Examples
/// Let us create some value but we will greet the user from the threads
/// ```
/// # use lattice_qcd_rs::thread::run_pool_parallel_with_initialisation_mutable;
/// # use std::error::Error;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let iter = 0_u128..100000_u128;
/// let c = 5_u128;
/// // we could have put 4 inside the closure but this demonstrate how to use common data
/// let result = run_pool_parallel_with_initialisation_mutable(
///     iter,
///     &c,
///     &|has_greeted: &mut bool, i, c| {
///         if !*has_greeted {
///             *has_greeted = true;
///             println!("Hello from the thread");
///         }
///         i * i * c
///     },
///     || false,
///     4,
///     100000,
/// )?;
/// # Ok(())
/// # }
/// ```
/// will print "Hello from the thread" four times.
///
/// Another useful application is to use an rng
/// ```
/// extern crate rand;
/// extern crate rand_distr;
/// use lattice_qcd_rs::field::Su3Adjoint;
/// use lattice_qcd_rs::lattice::LatticeCyclique;
/// use lattice_qcd_rs::thread::run_pool_parallel_with_initialisation_mutable;
/// # use std::error::Error;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let l = LatticeCyclique::<4>::new(1_f64, 4)?;
/// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
/// let result = run_pool_parallel_with_initialisation_mutable(
///     l.get_links(),
///     &distribution,
///     &|rng, _, d| Su3Adjoint::random(rng, d).to_su3(),
///     rand::thread_rng,
///     4,
///     l.number_of_canonical_links_space(),
/// )?;
/// # Ok(())
/// # }
/// ```
#[allow(clippy::needless_return)] // for lisibiliy
#[allow(clippy::semicolon_if_nothing_returned)] // I actually want to retun a never in the future
pub fn run_pool_parallel_with_initialisation_mutable<Key, Data, CommonData, InitData, F, FInit>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    closure_init: FInit,
    number_of_thread: usize,
    capacity: usize,
) -> Result<HashMap<Key, Data>, ThreadAnyError>
where
    CommonData: Sync,
    Key: Eq + Hash + Send + Clone + Sync,
    Data: Send,
    F: Sync + Clone + Fn(&mut InitData, &Key, &CommonData) -> Data,
    FInit: Send + Clone + FnOnce() -> InitData,
{
    if number_of_thread == 0 {
        return Err(ThreadAnyError::ThreadNumberIncorect);
    }
    else if number_of_thread == 1 {
        let mut hash_map = HashMap::<Key, Data>::with_capacity(capacity);
        let mut init_data = closure_init();
        for i in iter {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                hash_map.insert(i.clone(), closure(&mut init_data, &i, common_data))
            }))
            .map_err(|err| ThreadAnyError::Panic(vec![err]))?;
        }
        return Ok(hash_map);
    }
    else {
        let result = thread::scope(|s| {
            let mutex_iter = Arc::new(Mutex::new(iter));
            let mut threads = Vec::with_capacity(number_of_thread);
            let (result_tx, result_rx) = mpsc::channel::<(Key, Data)>();
            for _ in 0..number_of_thread {
                let iter_clone = Arc::clone(&mutex_iter);
                let transmitter = result_tx.clone();
                let closure_init_clone = closure_init.clone();
                let handel = s.spawn(move |_| {
                    let mut init_data = closure_init_clone();
                    loop {
                        let val = iter_clone.lock().unwrap().next();
                        match val {
                            Some(i) => transmitter
                                .send((i.clone(), closure(&mut init_data, &i, common_data)))
                                .unwrap(),
                            None => break,
                        }
                    }
                });
                threads.push(handel);
            }
            // we drop channel so we can properly assert if they are closed
            drop(result_tx);
            let mut hash_map = HashMap::<Key, Data>::with_capacity(capacity);
            for message in result_rx {
                let (key, data) = message;
                hash_map.insert(key, data);
            }

            let panics = threads
                .into_iter()
                .map(|handel| handel.join())
                .filter_map(|res| res.err())
                .collect::<Vec<_>>();
            if !panics.is_empty() {
                return Err(ThreadAnyError::Panic(panics));
            }

            Ok(hash_map)
        })
        .unwrap_or_else(|err| {
            if err
                .downcast_ref::<Vec<Box<dyn Any + 'static + Send>>>()
                .is_some()
            {
                unreachable!("a failing handle is not joined")
            }
            unreachable!("main thread panicked")
        });
        return result;
    }
}

/// run jobs in parallel. Similar to [`run_pool_parallel`] but return a vector.
///
/// Now a reference to the lattice must be given and `key` must implement the trait
/// [`super::lattice::LatticeElementToIndex`].
/// [`super::lattice::LatticeElementToIndex::to_index`] will be use to insert the data inside the vector.
/// While computing because the thread can operate out of order, fill the data not yet computed by `default_data`
/// `capacity` is used to determine the capacity of the [`std::vec::Vec`] upon initiation
/// (see [`std::vec::Vec::with_capacity`]).
///
/// # Errors
/// Returns [`ThreadAnyError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadAnyError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Example
/// ```
/// use lattice_qcd_rs::field::Su3Adjoint;
/// use lattice_qcd_rs::lattice::{LatticeCyclique, LatticeElementToIndex, LatticePoint};
/// use lattice_qcd_rs::thread::run_pool_parallel_vec;
/// # use std::error::Error;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let l = LatticeCyclique::<4>::new(1_f64, 4)?;
/// let c = 5_usize;
/// let result = run_pool_parallel_vec(
///     l.get_points(),
///     &c,
///     &|i: &LatticePoint<4>, c: &usize| i[0] * c,
///     4,
///     l.number_of_canonical_links_space(),
///     &l,
///     &0,
/// )?;
/// let point = LatticePoint::new([3, 0, 5, 0].into());
/// assert_eq!(result[point.to_index(&l)], point[0] * c);
/// # Ok(())
/// # }
/// ```
pub fn run_pool_parallel_vec<Key, Data, CommonData, F, const D: usize>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    number_of_thread: usize,
    capacity: usize,
    l: &LatticeCyclique<D>,
    default_data: &Data,
) -> Result<Vec<Data>, ThreadAnyError>
where
    CommonData: Sync,
    Key: Eq + Send + Clone + Sync + LatticeElementToIndex<D>,
    Data: Send + Clone,
    F: Sync + Clone + Fn(&Key, &CommonData) -> Data,
{
    run_pool_parallel_vec_with_initialisation_mutable(
        iter,
        common_data,
        &|_, key, common| closure(key, common),
        &|| (),
        number_of_thread,
        capacity,
        l,
        default_data,
    )
}

// TODO convert closure for conversion key -> usize

/// run jobs in parallel. Similar to [`run_pool_parallel_vec`] but with initiation.
///
/// # Errors
/// Returns [`ThreadAnyError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadAnyError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Examples
/// Let us create some value but we will greet the user from the threads
/// ```
/// use lattice_qcd_rs::lattice::{LatticeCyclique, LatticeElementToIndex, LatticePoint};
/// use lattice_qcd_rs::thread::run_pool_parallel_vec_with_initialisation_mutable;
/// # use std::error::Error;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let l = LatticeCyclique::<4>::new(1_f64, 25)?;
/// let iter = l.get_points();
/// let c = 5_usize;
/// // we could have put 4 inside the closure but this demonstrate how to use common data
/// let result = run_pool_parallel_vec_with_initialisation_mutable(
///     iter,
///     &c,
///     &|has_greeted: &mut bool, i: &LatticePoint<4>, c: &usize| {
///         if !*has_greeted {
///             *has_greeted = true;
///             println!("Hello from the thread");
///         }
///         i[0] * c
///     },
///     || false,
///     4,
///     100000,
///     &l,
///     &0,
/// )?;
/// # Ok(())
/// # }
/// ```
/// will print "Hello from the thread" four times.
///
/// Another useful application is to use an rng
/// ```
/// extern crate rand;
/// extern crate rand_distr;
/// extern crate nalgebra;
/// use lattice_qcd_rs::field::Su3Adjoint;
/// use lattice_qcd_rs::lattice::LatticeCyclique;
/// use lattice_qcd_rs::thread::run_pool_parallel_vec_with_initialisation_mutable;
/// # use std::error::Error;
///
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let l = LatticeCyclique::<4>::new(1_f64, 4)?;
/// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
/// let result = run_pool_parallel_vec_with_initialisation_mutable(
///     l.get_links(),
///     &distribution,
///     &|rng, _, d| Su3Adjoint::random(rng, d).to_su3(),
///     rand::thread_rng,
///     4,
///     l.number_of_canonical_links_space(),
///     &l,
///     &nalgebra::Matrix3::<nalgebra::Complex<f64>>::zeros(),
/// )?;
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_return)] // for lisibiliy
#[allow(clippy::semicolon_if_nothing_returned)] // I actually want to retun a never in the future
pub fn run_pool_parallel_vec_with_initialisation_mutable<
    Key,
    Data,
    CommonData,
    InitData,
    F,
    FInit,
    const D: usize,
>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    closure_init: FInit,
    number_of_thread: usize,
    capacity: usize,
    l: &LatticeCyclique<D>,
    default_data: &Data,
) -> Result<Vec<Data>, ThreadAnyError>
where
    CommonData: Sync,
    Key: Eq + Send + Clone + Sync,
    Data: Send + Clone,
    F: Sync + Clone + Fn(&mut InitData, &Key, &CommonData) -> Data,
    FInit: Send + Clone + FnOnce() -> InitData,
    Key: LatticeElementToIndex<D>,
{
    if number_of_thread == 0 {
        return Err(ThreadAnyError::ThreadNumberIncorect);
    }
    else if number_of_thread == 1 {
        let mut vec = Vec::<Data>::with_capacity(capacity);
        let mut init_data = closure_init();
        for i in iter {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                insert_in_vec(
                    &mut vec,
                    i.clone().to_index(l),
                    closure(&mut init_data, &i, common_data),
                    default_data,
                );
            }))
            .map_err(|err| ThreadAnyError::Panic(vec![err]))?;
        }
        return Ok(vec);
    }
    else {
        let result = thread::scope(|s| {
            // I try to put the thread creation in a function but the life time annotation were a mess.
            // I did not manage to make it working.
            let mutex_iter = Arc::new(Mutex::new(iter));
            let mut threads = Vec::with_capacity(number_of_thread);
            let (result_tx, result_rx) = mpsc::channel::<(Key, Data)>();
            for _ in 0..number_of_thread {
                let iter_clone = Arc::clone(&mutex_iter);
                let transmitter = result_tx.clone();
                let closure_init_clone = closure_init.clone();
                let handel = s.spawn(move |_| {
                    let mut init_data = closure_init_clone();
                    loop {
                        let val = iter_clone.lock().unwrap().next();
                        match val {
                            Some(i) => transmitter
                                .send((i.clone(), closure(&mut init_data, &i, common_data)))
                                .unwrap(),
                            None => break,
                        }
                    }
                });
                threads.push(handel);
            }
            // we drop channel so we can proprely assert if they are closed
            drop(result_tx);
            let mut vec = Vec::<Data>::with_capacity(capacity);
            for message in result_rx {
                let (key, data) = message;
                insert_in_vec(&mut vec, key.to_index(l), data, default_data);
            }

            let panics = threads
                .into_iter()
                .map(|handel| handel.join())
                .filter_map(|res| res.err())
                .collect::<Vec<_>>();
            if !panics.is_empty() {
                return Err(ThreadAnyError::Panic(panics));
            }

            Ok(vec)
        })
        .unwrap_or_else(|err| {
            if err
                .downcast_ref::<Vec<Box<dyn Any + 'static + Send>>>()
                .is_some()
            {
                unreachable!("a failing handle is not joined")
            }
            unreachable!("main thread panicked")
        });
        return result;
    }
}

/// Try setting the value inside the vec at position `pos`. If the position is not the array,
/// build the array with default value up to `pos - 1` and insert data at `pos`.
///
/// # Example
/// ```
/// # use lattice_qcd_rs::thread::insert_in_vec;
/// use std::vec::Vec;
///
/// let mut vec = vec![];
/// insert_in_vec(&mut vec, 0, 1, &0);
/// assert_eq!(vec, vec![1]);
/// insert_in_vec(&mut vec, 3, 9, &0);
/// assert_eq!(vec, vec![1, 0, 0, 9]);
/// insert_in_vec(&mut vec, 5, 10, &1);
/// assert_eq!(vec, vec![1, 0, 0, 9, 1, 10]);
/// insert_in_vec(&mut vec, 1, 3, &1);
/// assert_eq!(vec, vec![1, 3, 0, 9, 1, 10]);
/// ```
pub fn insert_in_vec<Data>(vec: &mut Vec<Data>, pos: usize, data: Data, default_data: &Data)
where
    Data: Clone,
{
    if pos < vec.len() {
        vec[pos] = data;
    }
    else {
        for _ in vec.len()..pos {
            vec.push(default_data.clone());
        }
        vec.push(data);
    }
}

/// Run a parallel pool using external crate [`rayon`].
///
/// # Example.
/// ```
/// # use lattice_qcd_rs::thread::run_pool_parallel_rayon;
/// let iter = 0..1000;
/// let c = 5;
/// let result = run_pool_parallel_rayon(iter, &c, |i, c1| i * i * c1);
/// assert_eq!(result[687], 687 * 687 * c);
/// assert_eq!(result[10], 10 * 10 * c);
/// ```
/// # Panic.
/// panic if the closure panic at any point during the evalutation
/// ```should_panic
/// # use lattice_qcd_rs::thread::run_pool_parallel_rayon;
/// let iter = 0..10;
/// let result = run_pool_parallel_rayon(iter, &(), |_, _| panic!("message"));
/// ```
pub fn run_pool_parallel_rayon<Key, Data, CommonData, F>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: F,
) -> Vec<Data>
where
    CommonData: Sync,
    Key: Eq + Send,
    Data: Send,
    F: Sync + Fn(&Key, &CommonData) -> Data,
{
    iter.collect::<Vec<Key>>()
        .into_par_iter()
        .map(|el| closure(&el, common_data))
        .collect()
}

#[cfg(test)]
mod test {
    use std::error::Error;

    use super::*;
    use crate::error::ImplementationError;

    #[test]
    fn thread_error() {
        assert_eq!(
            format!("{}", ThreadAnyError::ThreadNumberIncorect),
            "number of thread is incorrect"
        );
        assert!(
            format!("{}", ThreadAnyError::Panic(vec![Box::new(())])).contains("a thread panicked")
        );
        assert!(
            format!("{}", ThreadAnyError::Panic(vec![Box::new("message 1")])).contains("message 1")
        );
        assert!(format!("{}", ThreadAnyError::Panic(vec![])).contains("0 thread panicked"));

        assert!(ThreadAnyError::ThreadNumberIncorect.source().is_none());
        assert!(ThreadAnyError::Panic(vec![Box::new(())]).source().is_none());
        assert!(
            ThreadAnyError::Panic(vec![Box::new(ImplementationError::Unreachable)])
                .source()
                .is_none()
        );
        assert!(ThreadAnyError::Panic(vec![Box::new("test")])
            .source()
            .is_none());
        // -------
        assert_eq!(
            format!("{}", ThreadError::ThreadNumberIncorect),
            "number of thread is incorrect"
        );
        assert!(format!("{}", ThreadError::Panic(vec![None])).contains("a thread panicked"));
        assert!(format!("{}", ThreadError::Panic(vec![None, None])).contains("2 threads panicked"));
        assert!(format!(
            "{}",
            ThreadError::Panic(vec![Some("message 1".to_string())])
        )
        .contains("message 1"));
        assert!(format!("{}", ThreadError::Panic(vec![])).contains("0 thread panicked"));

        assert!(ThreadError::ThreadNumberIncorect.source().is_none());
        assert!(ThreadError::Panic(vec![None]).source().is_none());
        assert!(ThreadError::Panic(vec![Some("".to_string())])
            .source()
            .is_none());
        assert!(ThreadError::Panic(vec![Some("test".to_string())])
            .source()
            .is_none());
        //---------------

        let error = ThreadAnyError::Panic(vec![
            Box::new(()),
            Box::new("t1"),
            Box::new("t2".to_string()),
        ]);
        let error2 = ThreadAnyError::Panic(vec![
            Box::new(""),
            Box::new("t1".to_string()),
            Box::new("t2".to_string()),
        ]);
        let error3 = ThreadError::Panic(vec![None, Some("t1".to_string()), Some("t2".to_string())]);
        assert_eq!(ThreadError::from(error), error3);
        assert_eq!(ThreadAnyError::from(error3).to_string(), error2.to_string());

        let error = ThreadAnyError::ThreadNumberIncorect;
        let error2 = ThreadError::ThreadNumberIncorect;
        assert_eq!(ThreadError::from(error), error2);
        let error = ThreadAnyError::ThreadNumberIncorect;
        assert_eq!(ThreadAnyError::from(error2).to_string(), error.to_string());
    }
}
