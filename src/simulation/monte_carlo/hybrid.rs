//! Combine multiple Monte Carlo methods.
//!
//! this module present different ways to combine multiple method
//!
//! # Example
//! ```
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use lattice_qcd_rs::simulation::{HeatBathSweep, LatticeState, LatticeStateDefault, OverrelaxationSweepReverse, HybridMethodVec};
//! use rand::SeedableRng;
//!
//! let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
//! let mut heat_bath = HeatBathSweep::new(rng);
//! let mut overrelax = OverrelaxationSweepReverse::default();
//! let mut hybrid = HybridMethodVec::with_capacity(2);
//! hybrid.push_method(&mut heat_bath);
//! hybrid.push_method(&mut overrelax);
//!
//! let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 8_f64, 4)?; // 1_f64 : size, 8_f64: beta, 4 number of points.
//! for _ in 0..2 {
//!     state = state.monte_carlo_step(&mut hybrid)?;
//!     // operation to track the progress or the evolution
//! }
//! // operation at the end of the simulation
//! #     Ok(())
//! # }
//! ```

use core::fmt::{Debug, Display};
use std::error::Error;
use std::marker::PhantomData;
use std::vec::Vec;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{super::state::LatticeState, MonteCarlo};

/// Error given by [`HybridMethodVec`]
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum HybridMethodVecError<Error> {
    /// An Error coming from one of the method.
    ///
    /// the first field usize gives the position of the method giving the error
    Error(usize, Error),
    /// No method founds, give back the ownership of the state.
    NoMethod,
}

impl<Error: Display> Display for HybridMethodVecError<Error> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoMethod => write!(f, "no monte carlo method"),
            Self::Error(index, error) => {
                write!(f, "error during integration step {}: {}", index, error)
            }
        }
    }
}

impl<E: Display + Debug + Error + 'static> Error for HybridMethodVecError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NoMethod => None,
            Self::Error(_, error) => Some(error),
        }
    }
}

/// Adaptor used to convert the error to another type. It is intended to use with [`HybridMethodVec`].
#[derive(PartialEq, Eq, Debug)]
pub struct AdaptorMethodError<'a, MC, State, ErrorBase, Error, const D: usize>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
{
    data: &'a mut MC,
    _phantom: PhantomData<(&'a State, &'a ErrorBase, &'a Error)>,
}

impl<'a, MC, State, ErrorBase, Error, const D: usize>
    AdaptorMethodError<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
{
    /// Create the Self using a mutable reference.
    pub fn new(data: &'a mut MC) -> Self {
        Self {
            data,
            _phantom: PhantomData,
        }
    }

    /// Getter for the reference hold by self.
    pub fn data_mut(&mut self) -> &mut MC {
        self.data
    }

    /// Getter for the reference hold by self.
    pub const fn data(&self) -> &MC {
        self.data
    }
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> AsMut<MC>
    for AdaptorMethodError<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
{
    fn as_mut(&mut self) -> &mut MC {
        self.data_mut()
    }
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> AsRef<MC>
    for AdaptorMethodError<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
{
    fn as_ref(&self) -> &MC {
        self.data()
    }
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> MonteCarlo<State, D>
    for AdaptorMethodError<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
{
    type Error = Error;

    #[inline]
    fn next_element(&mut self, state: State) -> Result<State, Self::Error> {
        self.data.next_element(state).map_err(|err| err.into())
    }
}

/// hybrid method that combine multiple methods. It requires that all methods return the same error.
/// You can use [`AdaptorMethodError`] to convert the error.
/// If you want type with different error you can use [`HybridMethodCouple`].
///
/// # Example
/// see level module example [`super`].
pub struct HybridMethodVec<'a, State, E, const D: usize>
where
    State: LatticeState<D>,
{
    methods: Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>,
}

impl<'a, State, E, const D: usize> HybridMethodVec<'a, State, E, D>
where
    State: LatticeState<D>,
{
    getter!(
        /// get the methods
        pub,
        methods,
        Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>
    );

    /// Create an empty Self.
    pub fn new_empty() -> Self {
        Self {
            methods: Vec::new(),
        }
    }

    /// Create an empty Self where the vector is preallocated for `capacity` element.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            methods: Vec::with_capacity(capacity),
        }
    }

    /// Create a new Self from a list of [`MonteCarlo`]
    pub fn new(methods: Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>) -> Self {
        Self { methods }
    }

    /// Get a mutable reference to the methods used,
    pub fn methods_mut(&mut self) -> &mut Vec<&'a mut dyn MonteCarlo<State, D, Error = E>> {
        &mut self.methods
    }

    /// Add a method at the end.
    pub fn push_method(&mut self, mc_ref: &'a mut dyn MonteCarlo<State, D, Error = E>) {
        self.methods.push(mc_ref);
    }

    /// Remove a method at the end an returns it. Return None if the methods is empty.
    pub fn pop_method(&mut self) -> Option<&'a mut dyn MonteCarlo<State, D, Error = E>> {
        self.methods.pop()
    }

    /// Get the number of methods
    pub fn len(&self) -> usize {
        self.methods.len()
    }

    /// Return wether the number is zero.
    pub fn is_empty(&self) -> bool {
        self.methods.is_empty()
    }
}

impl<'a, State, E, const D: usize> AsRef<Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>>
    for HybridMethodVec<'a, State, E, D>
where
    State: LatticeState<D>,
{
    fn as_ref(&self) -> &Vec<&'a mut dyn MonteCarlo<State, D, Error = E>> {
        self.methods()
    }
}

impl<'a, State, E, const D: usize> AsMut<Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>>
    for HybridMethodVec<'a, State, E, D>
where
    State: LatticeState<D>,
{
    fn as_mut(&mut self) -> &mut Vec<&'a mut dyn MonteCarlo<State, D, Error = E>> {
        self.methods_mut()
    }
}

impl<'a, State, E, const D: usize> Default for HybridMethodVec<'a, State, E, D>
where
    State: LatticeState<D>,
{
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<'a, State, E, const D: usize> MonteCarlo<State, D> for HybridMethodVec<'a, State, E, D>
where
    State: LatticeState<D>,
{
    type Error = HybridMethodVecError<E>;

    #[inline]
    fn next_element(&mut self, mut state: State) -> Result<State, Self::Error> {
        if self.methods.is_empty() {
            return Err(HybridMethodVecError::NoMethod);
        }
        for (index, m) in &mut self.methods.iter_mut().enumerate() {
            let result = state.monte_carlo_step(*m);
            match result {
                Ok(new_state) => state = new_state,
                Err(error) => return Err(HybridMethodVecError::Error(index, error)),
            }
        }
        Ok(state)
    }
}

/// Error given by [`HybridMethodCouple`]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum HybridMethodCoupleError<Error1, Error2> {
    /// First method gave an error
    ErrorFirst(Error1),
    /// Second method gave an error
    ErrorSecond(Error2),
}

impl<Error1: Display, Error2: Display> Display for HybridMethodCoupleError<Error1, Error2> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ErrorFirst(error) => write!(f, "error during integration step 1: {}", error),
            Self::ErrorSecond(error) => write!(f, "error during integration step 2: {}", error),
        }
    }
}

impl<Error1: Display + Error + 'static, Error2: Display + Error + 'static> Error
    for HybridMethodCoupleError<Error1, Error2>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ErrorFirst(error) => Some(error),
            Self::ErrorSecond(error) => Some(error),
        }
    }
}

/// This method can combine any two methods. The down side is that it can be very verbose to write
/// Couples for a large number of methods.
///
/// # Example
/// ```
/// # use std::error::Error;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// use lattice_qcd_rs::simulation::{HeatBathSweep, LatticeState, LatticeStateDefault, OverrelaxationSweepReverse, HybridMethodCouple};
/// use rand::SeedableRng;
///
/// let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
/// let heat_bath = HeatBathSweep::new(rng);
/// let overrelax = OverrelaxationSweepReverse::default();
/// let mut couple = HybridMethodCouple::new(heat_bath,overrelax);
///
/// let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 8_f64, 4)?; // 1_f64 : size, 8_f64: beta, 4 number of points.
/// for _ in 0..2 {
///     state = state.monte_carlo_step(&mut couple)?;
///     // operation to track the progress or the evolution
/// }
/// // operation at the end of the simulation
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HybridMethodCouple<MC1, Error1, MC2, Error2, State, const D: usize>
where
    MC1: MonteCarlo<State, D, Error = Error1>,
    MC2: MonteCarlo<State, D, Error = Error2>,
    State: LatticeState<D>,
{
    method_1: MC1,
    method_2: MC2,
    _phantom: PhantomData<(State, Error1, Error2)>,
}

impl<MC1, Error1, MC2, Error2, State, const D: usize>
    HybridMethodCouple<MC1, Error1, MC2, Error2, State, D>
where
    MC1: MonteCarlo<State, D, Error = Error1>,
    MC2: MonteCarlo<State, D, Error = Error2>,
    State: LatticeState<D>,
{
    getter!(
        /// get the first method
        pub const,
        method_1,
        MC1
    );

    getter!(
        /// get the second method
        pub const,
        method_2,
        MC2
    );

    /// Create a new Self from two methods
    pub const fn new(method_1: MC1, method_2: MC2) -> Self {
        Self {
            method_1,
            method_2,
            _phantom: PhantomData,
        }
    }

    /// Deconstruct the structure ang gives back both methods
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn deconstruct(self) -> (MC1, MC2) {
        (self.method_1, self.method_2)
    }
}

impl<MC1, Error1, MC2, Error2, State, const D: usize> MonteCarlo<State, D>
    for HybridMethodCouple<MC1, Error1, MC2, Error2, State, D>
where
    MC1: MonteCarlo<State, D, Error = Error1>,
    MC2: MonteCarlo<State, D, Error = Error2>,
    State: LatticeState<D>,
{
    type Error = HybridMethodCoupleError<Error1, Error2>;

    #[inline]
    fn next_element(&mut self, mut state: State) -> Result<State, Self::Error> {
        state = state
            .monte_carlo_step(&mut self.method_1)
            .map_err(HybridMethodCoupleError::ErrorFirst)?;
        state
            .monte_carlo_step(&mut self.method_2)
            .map_err(HybridMethodCoupleError::ErrorSecond)
    }
}

/// Combine three methods.
pub type HybridMethodTriple<MC1, Error1, MC2, Error2, MC3, Error3, State, const D: usize> =
    HybridMethodCouple<
        HybridMethodCouple<MC1, Error1, MC2, Error2, State, D>,
        HybridMethodCoupleError<Error1, Error2>,
        MC3,
        Error3,
        State,
        D,
    >;

/// Error returned by [`HybridMethodTriple`].
pub type HybridMethodTripleError<Error1, Error2, Error3> =
    HybridMethodCoupleError<HybridMethodCoupleError<Error1, Error2>, Error3>;

/// Combine four methods.
pub type HybridMethodQuadruple<
    MC1,
    Error1,
    MC2,
    Error2,
    MC3,
    Error3,
    MC4,
    Error4,
    State,
    const D: usize,
> = HybridMethodCouple<
    HybridMethodTriple<MC1, Error1, MC2, Error2, MC3, Error3, State, D>,
    HybridMethodTripleError<Error1, Error2, Error3>,
    MC4,
    Error4,
    State,
    D,
>;

/// Error returned by [`HybridMethodQuadruple`].
pub type HybridMethodQuadrupleError<Error1, Error2, Error3, Error4> =
    HybridMethodCoupleError<HybridMethodTripleError<Error1, Error2, Error3>, Error4>;

/// Combine four methods.
pub type HybridMethodQuintuple<
    MC1,
    Error1,
    MC2,
    Error2,
    MC3,
    Error3,
    MC4,
    Error4,
    MC5,
    Error5,
    State,
    const D: usize,
> = HybridMethodCouple<
    HybridMethodQuadruple<MC1, Error1, MC2, Error2, MC3, Error3, MC4, Error4, State, D>,
    HybridMethodQuadrupleError<Error1, Error2, Error3, Error4>,
    MC5,
    Error5,
    State,
    D,
>;

/// Error returned by [`HybridMethodQuintuple`].
pub type HybridMethodQuintupleError<Error1, Error2, Error3, Error4, Error5> =
    HybridMethodCoupleError<HybridMethodQuadrupleError<Error1, Error2, Error3, Error4>, Error5>;
