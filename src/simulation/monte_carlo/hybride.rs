
//! Combine multiple Monte Carlo methods.
//!
//! this module present different ways to combine multiple method

use super::{
    MonteCarlo,
    super::{
        super::{
            lattice::{
                Direction,
                DirectionList,
            },
        },
        state::{
            LatticeState,
        },
    },
};
use std::vec::Vec;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::error::Error;
use core::fmt::{Display, Debug};


/// Error given by [`HybrideMethodeVec`]
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
pub enum HybrideMethodeVecError<Error> {
    /// An Error comming from one of the method.
    ///
    /// the first field usize gives the position of the method giving the error
    Error(usize, Error),
    /// No method founds, give back the ownership of the state.
    NoMethod,
}

impl<Error: Display> Display for HybrideMethodeVecError<Error> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoMethod => write!(f, "error: no Monte Carlo method"),
            Self::Error(index, error) => write!(f, "error during intgration step {}: {}",index, error),
        }
    }
}


impl<E: Display + Debug + Error + 'static> Error for HybrideMethodeVecError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NoMethod => None,
            Self::Error(_, error) => Some(error),
        }
    }
}


/// Adaptator used to convert the error to another type. It is intented to use with [`HybrideMethodeVec`].
#[derive(PartialEq, Eq, Debug)]
pub struct AdaptatorErrorMethod<'a, MC, State, ErrorBase, Error, const D: usize>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    data: &'a mut MC,
    _phantom : PhantomData<(&'a State, &'a ErrorBase, &'a Error)>,
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> AdaptatorErrorMethod<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    /// Create the Self using a mutable reference
    pub fn new(data: &'a mut MC) -> Self {
        Self{data, _phantom: PhantomData}
    }
    
    /// getter for the reference holded by self
    pub fn data(&'a mut self) -> &'a mut MC {
        self.data
    }
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> Deref for AdaptatorErrorMethod<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    type Target = MC;
    
    fn deref(&self) -> &Self::Target {
       self.data
   }
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> DerefMut for AdaptatorErrorMethod<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
       self.data
   }
}

impl<'a, MC, State, ErrorBase, Error, const D: usize> MonteCarlo<State, D> for AdaptatorErrorMethod<'a, MC, State, ErrorBase, Error, D>
where
    MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    type Error = Error;
    
    #[inline]
    fn get_next_element(&mut self, state: State) -> Result<State, Self::Error> {
        self.data.get_next_element(state).map_err(|err| err.into())
    }
}

/// hybride methode that combine multiple methodes. It requires that all methods return the same error.
/// You can use [`AdaptatorErrorMethod`] to convert the error.
/// If you want type with different error you can use [`HybrideMethodeCouple`].
pub struct HybrideMethodeVec<'a, State, E, const D: usize>
where
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    methods: Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>
}

impl<'a, State, E, const D: usize> HybrideMethodeVec<'a, State, E, D>
where
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    /// Create an empty Self.
    pub fn new_empty() -> Self {
        Self {methods: vec![]}
    }
    
    /// Create an empty Self where the vector is preallocated for `capacity` element.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {methods: Vec::with_capacity(capacity)}
    }
    
    /// Create a new Self from a list of [`MonteCarlo`]
    pub fn new(methods: Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>) -> Self {
        Self {methods}
    }
    
    getter!(
        /// get the methods
        methods, Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>
    );
    
    /// Get a mutable reference to the methodes used,
    pub fn methods_mut(&mut self) -> &mut Vec<&'a mut dyn MonteCarlo<State, D, Error = E>> {
        &mut self.methods
    }
    
    /// Add a methode at the end.
    pub fn push_method(&mut self, mc_ref: &'a mut dyn MonteCarlo<State, D, Error = E>) {
        self.methods.push(mc_ref);
    }
    
    /// Remove a methode at the end an returns it. Return None if the methodes is empty.
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

impl<'a, State, E, const D: usize> Default for HybrideMethodeVec<'a, State, E, D>
where
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<'a, State, E, const D: usize> MonteCarlo<State, D> for HybrideMethodeVec<'a, State, E, D>
where
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    type Error = HybrideMethodeVecError<E>;
    
    #[inline]
    fn get_next_element(&mut self, mut state: State) -> Result<State, Self::Error> {
        if self.methods.is_empty() {
            return Err(HybrideMethodeVecError::NoMethod);
        }
        for (index, m) in &mut self.methods.iter_mut().enumerate() {
            let result = state.monte_carlo_step(*m);
            match result {
                Ok(new_state) => state = new_state,
                Err(error) => return Err(HybrideMethodeVecError::Error(index, error))
            }
        }
        Ok(state)
    }
}

/// Error given by [`HybrideMethodeCouple`]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum HybrideMethodeCoupleError<Error1, Error2> {
    /// First method gave an error
    ErrorFirst(Error1),
    /// Second method gave an error
    ErrorSecond(Error2),
}

impl<Error1: Display, Error2: Display> Display for HybrideMethodeCoupleError<Error1, Error2> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ErrorFirst(error) => write!(f, "Error during intgration step 1: {}", error),
            Self::ErrorSecond(error) => write!(f, "Error during intgration step 2: {}", error),
        }
    }
}


impl<Error1: Display + Error + 'static, Error2: Display + Error + 'static> Error for HybrideMethodeCoupleError<Error1, Error2> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ErrorFirst(error) => Some(error),
            Self::ErrorSecond(error) => Some(error),
        }
    }
}

/// This method can combine any two methods. The down side is that it can be very verbose to write
/// Couples for a large number of methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HybrideMethodeCouple<MC1, Error1, MC2, Error2, State, const D: usize>
where
    MC1: MonteCarlo<State, D, Error = Error1>,
    MC2: MonteCarlo<State, D, Error = Error2>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    method_1: MC1,
    method_2: MC2,
    _phantom: PhantomData<(State, Error1, Error2)>
}

impl<MC1, Error1, MC2, Error2, State, const D: usize> HybrideMethodeCouple<MC1, Error1, MC2, Error2, State, D>
where
    MC1: MonteCarlo<State, D, Error = Error1>,
    MC2: MonteCarlo<State, D, Error = Error2>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    /// Create a new Self from two methods
    pub fn new(method_1: MC1, method_2: MC2) -> Self{
        Self{method_1, method_2, _phantom: PhantomData}
    }
    
    getter!(
        /// get the first method
        method_1, MC1
    );
    
    getter!(
        /// get the second method
        method_2, MC2
    );
    
    /// Deconstruct the structure ang gives back both methods
    pub fn deconstruct(self) -> (MC1, MC2) {
        (self.method_1, self.method_2)
    }
}


impl<MC1, Error1, MC2, Error2, State, const D: usize> MonteCarlo<State, D> for HybrideMethodeCouple<MC1, Error1, MC2, Error2, State, D>
where
    MC1: MonteCarlo<State, D, Error = Error1>,
    MC2: MonteCarlo<State, D, Error = Error2>,
    State: LatticeState<D>,
    Direction<D>: DirectionList,
{
    type Error = HybrideMethodeCoupleError<Error1, Error2>;
    
    #[inline]
    fn get_next_element(&mut self, mut state: State) -> Result<State, Self::Error> {
        state = state.monte_carlo_step(&mut self.method_1).map_err(HybrideMethodeCoupleError::ErrorFirst)?;
        state.monte_carlo_step(&mut self.method_2).map_err(HybrideMethodeCoupleError::ErrorSecond)
    }
}

/// Combine three methods.
pub type HybrideMethodeTriple<MC1, Error1, MC2, Error2, MC3, Error3, State, const D: usize> = HybrideMethodeCouple<HybrideMethodeCouple<MC1, Error1, MC2, Error2, State, D>, HybrideMethodeCoupleError<Error1, Error2>, MC3, Error3, State, D>;

/// Error returned by [`HybrideMethodeTriple`].
pub type HybrideMethodeTripleError<Error1, Error2, Error3> = HybrideMethodeCoupleError<HybrideMethodeCoupleError<Error1, Error2>, Error3>;

/// Combine four methods.
pub type HybrideMethodeQuadruple<MC1, Error1, MC2, Error2, MC3, Error3, MC4, Error4, State, const D: usize> = HybrideMethodeCouple<HybrideMethodeTriple<MC1, Error1, MC2, Error2, MC3, Error3, State, D>, HybrideMethodeTripleError<Error1, Error2, Error3>, MC4, Error4, State, D>;

/// Error returned by [`HybrideMethodeQuadruple`].
pub type HybrideMethodeQuadrupleError<Error1, Error2, Error3, Error4> = HybrideMethodeCoupleError<HybrideMethodeTripleError<Error1, Error2, Error3>, Error4>;


/// Combine four methods.
pub type HybrideMethodeQuintuple<MC1, Error1, MC2, Error2, MC3, Error3, MC4, Error4, MC5, Error5, State, const D: usize> = HybrideMethodeCouple<HybrideMethodeQuadruple<MC1, Error1, MC2, Error2, MC3, Error3, MC4, Error4, State, D>, HybrideMethodeQuadrupleError<Error1, Error2, Error3, Error4>, MC5, Error5, State, D>;

/// Error returned by [`HybrideMethodeQuintuple`].
pub type HybrideMethodeTripleQuintuple<Error1, Error2, Error3, Error4, Error5> = HybrideMethodeCoupleError<HybrideMethodeQuadrupleError<Error1, Error2, Error3, Error4>, Error5>;
