
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
    pub fn new(methods: Vec<&'a mut dyn MonteCarlo<State, D>>) -> Self {
        Self {methods}
    }
    
    getter!(methods, Vec<&'a mut dyn MonteCarlo<State, D>>);
    
    pub fn add_methods(&mut self, mc_ref: &'a mut dyn MonteCarlo<State, D>) {
        self.methods.push(mc_ref)
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
        Self::new(vec![])
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
        for m in &mut self.methods {
            state = state.monte_carlo_step(*m)?;
        }
        Ok(state)
    }
}
