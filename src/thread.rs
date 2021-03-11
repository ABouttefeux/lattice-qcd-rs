
//! tool for easy use of mutli threading.

use std::{
    collections::HashMap,
    iter::Iterator,
    sync::{
        Arc,
        Mutex,
        mpsc,
    },
    any::Any,
    hash::Hash,
    vec::Vec
};
use crossbeam::thread;
use super::lattice::{
    LatticeCyclique,
    LatticeElementToIndex,
    Direction,
    DirectionList,
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
};

/// Multithreading error.
#[derive(Debug)]
#[non_exhaustive]
pub enum ThreadError {
    /// Tried to run some jobs with 0 threads
    ThreadNumberIncorect,
    /// One of the thread panicked. inside the [`Box`] is the panic message.
    /// see [`run_pool_parallel`] example.
    Panic(Box<dyn Any + Send + 'static>),
}


impl core::fmt::Display for ThreadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ThreadNumberIncorect => write!(f, "Number of thread is incorrect"),
            Self::Panic(any) => write!(f, "a thread panicked with message {:?}", any),
        }
    }
}

macro_rules! implement_dyn_downcast{
    ($any:ident, $to:ident $(, $t:ty)*) => {
        $(
            let downcast_r = $any.downcast_ref::<$t>().map(|el| el as &dyn $to);
            if downcast_r.is_some() {
                return downcast_r;
            }
        )*
        return None;
    }
}

impl std::error::Error for ThreadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use std::error::Error;
        use super::error::{ImplementationError, Never, StateInitializationError, StateInitializationErrorThreaded};
        match self {
            Self::ThreadNumberIncorect => None,
            Self::Panic(any) => {
                implement_dyn_downcast!(any, Error, Never, ImplementationError, StateInitializationError, StateInitializationErrorThreaded);
            },
        }
    }
}


/// run jobs in parallel.
///
/// The pool of job is given by `iter`. the job is given by `closure` that have the form `|key,common_data| -> Data`.
/// `number_of_thread` determine the number of job done in parallel and should be greater than 0,
/// otherwise return [`ThreadError::ThreadNumberIncorect`].
/// `capacity` is used to determine the capacity of the [`HashMap`] upon initisation (see [`HashMap::with_capacity`])
///
/// # Errors
/// Returns [`ThreadError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Example
/// let us computes the value of `i^2 * c` for i in [2,9999] with 4 threads
/// ```
/// # use lattice_qcd_rs::thread::run_pool_parallel;
/// let iter = 2..10000;
/// let c = 5;
/// // we could have put 4 inside the closure but this demonstrate how to use common data
/// let result = run_pool_parallel(iter, &c, &|i, c| {i * i * c} , 4, 10000 - 2).unwrap();
/// assert_eq!(*result.get(&40).unwrap(), 40 * 40 * c);
/// assert_eq!(result.get(&1), None);
/// ```
/// In the next example a thread will panic, we demonstrate the return type.
/// ```should_panic
/// # use lattice_qcd_rs::thread::{run_pool_parallel, ThreadError};
/// let iter = 0..10;
/// let result = run_pool_parallel(iter, &(), &|_, _| {panic!("panic message")}, 4, 10);
/// result.unwrap(); // this propagate the panic.
/// ```
/// This give the following panic message
/// ```textrust
/// ---- src\thread.rs - thread::run_pool_parallel (line 51) stdout ----
/// Test executable failed (exit code 101).
///
/// stderr:
/// thread '<unnamed>' panicked at 'thread 'panic message', src\thread.rs:6:60
/// note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
/// <unnamed>' panicked at 'panic message', src\thread.rs:6:60
/// thread '<unnamed>' panicked at 'panic message', src\thread.rs:6:60
/// thread '<unnamed>' panicked at 'panic message', src\thread.rs:6:60
/// thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: Panic(Any)', src\thread.rs:7:8
/// ```
pub fn run_pool_parallel<Key, Data, CommonData, F>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    number_of_thread: usize,
    capacity: usize,
) -> Result<HashMap<Key, Data>, ThreadError>
    where //Builder: BuildHasher + Default,
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
/// Returns [`ThreadError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Examples
/// Let us create some value but we will greet the user from the threads
/// ```
/// # use lattice_qcd_rs::thread::run_pool_parallel_with_initialisation_mutable;
/// let iter = 0_u128..100000_u128;
/// let c = 5_u128;
/// // we could have put 4 inside the closure but this demonstrate how to use common data
/// let result = run_pool_parallel_with_initialisation_mutable(
///     iter,
///     &c,
///     &|has_greeted: &mut bool, i, c| {
///          if ! *has_greeted {
///              *has_greeted = true;
///              println!("Hello from the thread");
///          }
///          i * i * c
///     },
///     || {false},
///     4,
///     100000
/// ).unwrap();
/// ```
/// will print "Hello from the thread" four times.
///
/// Another useful application is to use an rng
/// ```
/// extern crate rand;
/// extern crate rand_distr;
/// use lattice_qcd_rs::thread::run_pool_parallel_with_initialisation_mutable;
/// use lattice_qcd_rs::lattice::LatticeCyclique;
/// use lattice_qcd_rs::field::Su3Adjoint;
/// use lattice_qcd_rs::dim::U4;
///
/// let l = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
/// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
/// let result = run_pool_parallel_with_initialisation_mutable(
///     l.get_links(),
///     &distribution,
///     &|rng, _, d| Su3Adjoint::random(rng, d).to_su3(),
///     rand::thread_rng,
///     4,
///     l.get_number_of_canonical_links_space(),
/// ).unwrap();
/// ```
#[allow(clippy::needless_return)] // for lisibiliy
pub fn run_pool_parallel_with_initialisation_mutable<Key, Data, CommonData, InitData, F, FInit>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    closure_init: FInit,
    number_of_thread: usize,
    capacity: usize,
) -> Result<HashMap<Key, Data>, ThreadError>
    where //Builder: BuildHasher + Default,
    CommonData: Sync,
    Key: Eq + Hash + Send + Clone + Sync,
    Data: Send,
    F: Sync + Clone + Fn(&mut InitData, &Key, &CommonData) -> Data,
    FInit: Send + Clone + FnOnce() -> InitData,
{
    if number_of_thread == 0 {
        return Err(ThreadError::ThreadNumberIncorect);
    }
    else if number_of_thread== 1{
        let mut hash_map = HashMap::<Key, Data>::with_capacity(capacity);
        let mut init_data = closure_init();
        for i in iter{
            hash_map.insert(i.clone(), closure(&mut init_data, &i, common_data));
        }
        return Ok(hash_map);
    }
    else {
        let result = thread::scope(|s| {
            let mutex_iter = Arc::new(Mutex::new(iter));
            let mut threads = vec![];
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
                            Some(i) => transmitter.send((i.clone(), closure(&mut init_data, &i, common_data))).unwrap(),
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
            for handel in threads {
                handel.join().map_err(|err| ThreadError::Panic(err) )?;
            }
            Ok(hash_map)
        }).map_err(|err| ThreadError::Panic(err))?;
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
/// Returns [`ThreadError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Example
/// ```
/// use lattice_qcd_rs::thread::run_pool_parallel_vec;
/// use lattice_qcd_rs::lattice::{LatticeCyclique, LatticeElementToIndex, LatticePoint};
/// use lattice_qcd_rs::field::Su3Adjoint;
/// use lattice_qcd_rs::dim::U4;
///
/// let l = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
/// let c = 5_usize;
/// let result = run_pool_parallel_vec(
///     l.get_points(),
///     &c,
///     &|i: &LatticePoint<_>, c: &usize| i[0] * c,
///     4,
///     l.get_number_of_canonical_links_space(),
///     &l,
///     &0,
/// ).unwrap();
/// let point = LatticePoint::new([3, 0, 5, 0].into());
/// assert_eq!(result[point.to_index(&l)], point[0] * c)
/// ```
pub fn run_pool_parallel_vec<Key, Data, CommonData, F, D>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    number_of_thread: usize,
    capacity: usize,
    l: &LatticeCyclique<D>,
    default_data: &Data,
) -> Result<Vec<Data>, ThreadError>
    where CommonData: Sync,
    Key: Eq + Send + Clone + Sync + LatticeElementToIndex<D>,
    Data: Send + Clone,
    F: Sync + Clone + Fn(&Key, &CommonData) -> Data,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    run_pool_parallel_vec_with_initialisation_mutable(
        iter,
        common_data,
        &|_, key, common| { closure(key, common)},
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
/// Returns [`ThreadError::ThreadNumberIncorect`] is the number of threads is 0.
/// Returns [`ThreadError::Panic`] if a thread panicked. Containt the panick message.
///
/// # Examples
/// Let us create some value but we will greet the user from the threads
/// ```
/// use lattice_qcd_rs::thread::run_pool_parallel_vec_with_initialisation_mutable;
/// use lattice_qcd_rs::lattice::{LatticeCyclique, LatticeElementToIndex, LatticePoint};
/// use lattice_qcd_rs::dim::U4;
/// let l = LatticeCyclique::<U4>::new(1_f64, 25).unwrap();
/// let iter = l.get_points();
/// let c = 5_usize;
/// // we could have put 4 inside the closure but this demonstrate how to use common data
/// let result = run_pool_parallel_vec_with_initialisation_mutable(
///     iter,
///     &c,
///     &|has_greeted: &mut bool, i: &LatticePoint<_>, c: &usize| {
///          if ! *has_greeted {
///              *has_greeted = true;
///              println!("Hello from the thread");
///          }
///          i[0] * c
///     },
///     || {false},
///     4,
///     100000,
///     &l,
///     &0,
/// ).unwrap();
/// ```
/// will print "Hello from the thread" four times.
///
/// Another useful application is to use an rng
/// ```
/// extern crate rand;
/// extern crate rand_distr;
/// extern crate nalgebra;
/// use lattice_qcd_rs::thread::run_pool_parallel_vec_with_initialisation_mutable;
/// use lattice_qcd_rs::lattice::LatticeCyclique;
/// use lattice_qcd_rs::field::Su3Adjoint;
/// use lattice_qcd_rs::dim::U4;
///
/// let l = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
/// let distribution = rand::distributions::Uniform::from(-1_f64..1_f64);
/// let result = run_pool_parallel_vec_with_initialisation_mutable(
///     l.get_links(),
///     &distribution,
///     &|rng, _, d| Su3Adjoint::random(rng, d).to_su3(),
///     rand::thread_rng,
///     4,
///     l.get_number_of_canonical_links_space(),
///     &l,
///     &nalgebra::Matrix3::<nalgebra::Complex<f64>>::zeros(),
/// ).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_return)] // for lisibiliy
pub fn run_pool_parallel_vec_with_initialisation_mutable<Key, Data, CommonData, InitData, F, FInit, D>(
    iter: impl Iterator<Item = Key> + Send,
    common_data: &CommonData,
    closure: &F,
    closure_init: FInit,
    number_of_thread: usize,
    capacity: usize,
    l: &LatticeCyclique<D>,
    default_data: &Data,
) -> Result<Vec<Data>, ThreadError>
    where CommonData: Sync,
    Key: Eq + Send + Clone + Sync,
    Data: Send + Clone,
    F: Sync + Clone + Fn(&mut InitData, &Key, &CommonData) -> Data,
    FInit: Send + Clone + FnOnce() -> InitData,
    Key: LatticeElementToIndex<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    na::VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    if number_of_thread == 0 {
        return Err(ThreadError::ThreadNumberIncorect);
    }
    else if number_of_thread== 1{
        let mut vec = Vec::<Data>::with_capacity(capacity);
        let mut init_data = closure_init();
        for i in iter{
            insert_in_vec(&mut vec, i.clone().to_index(l), closure(&mut init_data, &i, common_data), &default_data);
        }
        return Ok(vec);
    }
    else {
        let result = thread::scope(|s| {
            // I try to put the thread creation in a function but the life time annotation were a mess.
            // I did not manage to make it working.
            let mutex_iter = Arc::new(Mutex::new(iter));
            let mut threads = vec![];
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
                            Some(i) => transmitter.send((i.clone(), closure(&mut init_data, &i, common_data))).unwrap(),
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
                insert_in_vec(&mut vec, key.to_index(l), data, &default_data);
            }
            for handel in threads {
                handel.join().map_err(|err| ThreadError::Panic(err) )?;
            }
            Ok(vec)
        }).map_err(|err| ThreadError::Panic(err))?;
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
    where Data: Clone,
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
    where CommonData: Sync,
    Key: Eq + Send,
    Data: Send,
    F: Sync + Fn(&Key, &CommonData) -> Data,
{
    iter.collect::<Vec<Key>>().into_par_iter().map(|el| closure(&el, common_data)).collect()
}
