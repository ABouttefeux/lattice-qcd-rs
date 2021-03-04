
use super::{
    MonteCarlo,
    get_straple,
    super::{
        super::{
            Complex,
            CMatrix2,
            CMatrix3,
            statistics::HeatBathDistribution,
            field::{
                LinkMatrix,
            },
            su3,
            lattice::{
                LatticePoint,
                LatticeLinkCanonical,
                Direction,
                LatticeElementToIndex,
                LatticeLink,
                LatticeCyclique,
                DirectionList,
            }
        },
        state::{
            LatticeState,
            LatticeStateDefault,
        },
        SimulationError,
    },
};
use na::{
    DimName,
    DefaultAllocator,
    base::allocator::Allocator,
    VectorN,
    ComplexField,
};

/// Pseudo heat bath algorithm
pub struct OverrelaxationSweep<Rng>
    where Rng: rand::Rng,
{
    rng: Rng
}

impl<Rng> OverrelaxationSweep<Rng>
    where Rng: rand::Rng,
{
    
    pub fn new(rng: Rng) -> Self {
        Self {rng}
    }
    
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    pub fn rng(&mut self) -> &mut Rng {
        &mut self.rng
    }
    
    #[inline]
    fn get_potential_modif<D>(&mut self, state: &LatticeStateDefault<D>, link: &LatticeLinkCanonical<D>) -> na::Matrix3<Complex>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let link_matrix = state.link_matrix().get_matrix(&link.into(), state.lattice()).unwrap();
        let a = get_straple(state.link_matrix(), state.lattice(), link);
        
        todo!()
    }
    
    #[inline]
    fn get_next_element_default<D>(&mut self, mut state: LatticeStateDefault<D>) -> LatticeStateDefault<D>
        where D: DimName,
        DefaultAllocator: Allocator<usize, D>,
        na::VectorN<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = self.get_potential_modif(&state, &link);
            *state.get_link_mut(&link).unwrap() = potential_modif;
        });
        state
    }
}

impl<Rng, D> MonteCarlo<LatticeStateDefault<D>, D> for OverrelaxationSweep<Rng>
    where Rng: rand::Rng,
    D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault<D>) -> Result<LatticeStateDefault<D>, SimulationError>{
        Ok(self.get_next_element_default(state))
    }
}
