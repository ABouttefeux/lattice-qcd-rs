
//! Combine multiple methods.

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
        SimulationError,
    },
};
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
};
use std::vec::Vec;

/// hybride methode that combine multiple methodes
pub struct HybrideMethode<'a, State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    methods: Vec<&'a mut dyn MonteCarlo<State, D>>
}

impl<'a, State, D> HybrideMethode<'a, State, D>
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
    pub fn new(methods: Vec<&'a mut dyn MonteCarlo<State, D>>) -> Self {
        Self {methods}
    }
    
    getter!(
        /// get the methods
        methods, Vec<&'a mut dyn MonteCarlo<State, D>>
    );
    
    /// Get a mutable reference to the methodes used,
    pub fn methods_mut(&mut self) -> &mut Vec<&'a mut dyn MonteCarlo<State, D>> {
        &mut self.methods
    }
    
    /// Add a methode at the end.
    pub fn push_methods(&mut self, mc_ref: &'a mut dyn MonteCarlo<State, D>) {
        self.methods.push(mc_ref);
    }
    
    /// Remove a methode at the end an returns it. Return None if the methodes is empty.
    pub fn pop_method(&mut self) -> Option<&'a mut dyn MonteCarlo<State, D>> {
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

impl<'a, State, D> Default for HybrideMethode<'a, State, D>
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

impl<'a, State, D> MonteCarlo<State, D> for HybrideMethode<'a, State, D>
    where State: LatticeState<D>,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn get_next_element(&mut self, mut state: State) -> Result<State, SimulationError> {
        if self.methods.is_empty() {
            return Err(SimulationError::ZeroStep);
        }
        for m in &mut self.methods {
            state = state.monte_carlo_step(*m)?;
        }
        Ok(state)
    }
}
