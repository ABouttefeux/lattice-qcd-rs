
//! Combine multiple methods.

use super::{
    MonteCarlo,
    super::{
        super::{
            lattice::{
                Direction,
                DirectionList,
            },
            error::GetOwnedValue,
        },
        state::{
            LatticeState,
        },
    },
};
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
};
use std::vec::Vec;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::error::Error;
use core::fmt::{Display, Debug};


/// Error given by [`HybrideMethode`]
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Hash, Copy, Debug)]
pub enum HybrideMethodeError<Error, State> {
    /// An Error comming from one of the method.
    ///
    /// the first field usize gives the position of the method giving the error
    Error(usize, Error),
    /// No method founds, give back the ownership of the state.
    NoMethod(State),
}

impl<Error, State> HybrideMethodeError<Error, State> {
    /// Return the state if the error is [`HybrideMethodeError::NoMethod`].
    /// # Errors
    /// Return the error of [`HybrideMethodeError::Error`] otherwise.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn get_state(self) -> Result<State, Error>{
        match self {
            Self::Error(_, error) => Err(error),
            Self::NoMethod(state) => Ok(state),
        }
    }
}

impl<Error: Display, State> Display for HybrideMethodeError<Error, State> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HybrideMethodeError::NoMethod(_) => write!(f, "error: no Monte Carlo method"),
            HybrideMethodeError::Error(index, error) => write!(f, "error during intgration step {}: {}",index, error),
        }
    }
}


impl<E: Display + Debug + Error + 'static, State: Debug> Error for HybrideMethodeError<E, State> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            HybrideMethodeError::NoMethod(_) => None,
            HybrideMethodeError::Error(_, error) => Some(error),
        }
    }
}

impl<Error: GetOwnedValue<State>, State> GetOwnedValue<State> for HybrideMethodeError<Error, State> {
    fn get_owned_value(self) -> Option<State> {
        match self {
            HybrideMethodeError::NoMethod(state) => Some(state),
            HybrideMethodeError::Error(_, error) => error.get_owned_value(),
        }
    }
    
    fn get_ref_value(&self) -> Option<&State> {
        match self {
            HybrideMethodeError::NoMethod(state) => Some(state),
            HybrideMethodeError::Error(_, error) => error.get_ref_value(),
        }
    }
}


/// Adaptator used to convert the error to another type. It is intented to use with [`HybrideMethode`].
#[derive(PartialEq, Eq, Debug)]
pub struct AdaptatorErrorMethod<'a, MC, State, D, ErrorBase, Error>
    where MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    data: &'a mut MC,
    _phantom : PhantomData<(&'a State, &'a D, &'a ErrorBase, &'a Error)>,
}

impl<'a, MC, State, D, ErrorBase, Error> AdaptatorErrorMethod<'a, MC, State, D, ErrorBase, Error>
    where MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
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

impl<'a, MC, State, D, ErrorBase, Error> Deref for AdaptatorErrorMethod<'a, MC, State, D, ErrorBase, Error>
    where MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Target = MC;
    
    fn deref(&self) -> &Self::Target {
       self.data
   }
}

impl<'a, MC, State, D, ErrorBase, Error> DerefMut for AdaptatorErrorMethod<'a, MC, State, D, ErrorBase, Error>
    where MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
       self.data
   }
}

impl<'a, MC, State, D, ErrorBase, Error> MonteCarlo<State, D> for AdaptatorErrorMethod<'a, MC, State, D, ErrorBase, Error>
    where MC: MonteCarlo<State, D, Error = ErrorBase> + ?Sized,
    ErrorBase: Into<Error>,
    State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Error = Error;
    
    #[inline]
    fn get_next_element(&mut self, state: State) -> Result<State, Self::Error> {
        self.data.get_next_element(state).map_err(|err| err.into())
    }
}

/// hybride methode that combine multiple methodes
pub struct HybrideMethode<'a, State, D, E>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    methods: Vec<&'a mut dyn MonteCarlo<State, D, Error = E>>
}

impl<'a, State, D, E> HybrideMethode<'a, State, D, E>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
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

impl<'a, State, D, E> Default for HybrideMethode<'a, State, D, E>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn default() -> Self {
        Self::new_empty()
    }
}

impl<'a, State, D, E> MonteCarlo<State, D> for HybrideMethode<'a, State, D, E>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Error = HybrideMethodeError<E, State>;
    
    #[inline]
    fn get_next_element(&mut self, mut state: State) -> Result<State, Self::Error> {
        if self.methods.is_empty() {
            return Err(HybrideMethodeError::NoMethod(state));
        }
        for (index, m) in &mut self.methods.iter_mut().enumerate() {
            let result = state.monte_carlo_step(*m);
            match result {
                Ok(new_state) => state = new_state,
                Err(error) => return Err(HybrideMethodeError::Error(index, error))
            }
        }
        Ok(state)
    }
}
