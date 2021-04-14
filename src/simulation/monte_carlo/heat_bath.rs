
//! Pseudo heat bath methods

use super::{
    MonteCarlo,
    get_staple,
    super::{
        super::{
            Complex,
            CMatrix2,
            statistics::HeatBathDistribution,
            su3,
            lattice::{
                LatticeLinkCanonical,
                Direction,
                DirectionList,
            },
            error::Never,
            su2,
        },
        state::{
            LatticeState,
            LatticeStateDefault,
        },
    },
};
use na::{
    SVector,
    ComplexField,
};
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};


/// Pseudo heat bath algorithm
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HeatBathSweep<Rng>
    where Rng: rand::Rng,
{
    rng: Rng
}

impl<Rng> HeatBathSweep<Rng>
    where Rng: rand::Rng,
{
    /// Create a new Self form a rng.
    pub fn new(rng: Rng) -> Self {
        Self {rng}
    }
    
    /// Absorbe self and return the RNG as owned. It essentialy deconstruct the structure.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    /// Get a mutable reference to the rng.
    pub fn rng(&mut self) -> &mut Rng {
        &mut self.rng
    }
    
    #[inline]
    fn get_heat_bath_su2(&mut self, staple: CMatrix2, beta: f64) -> CMatrix2 {
        let staple_coeef = staple.determinant().real().sqrt();
        if staple_coeef.is_normal() {
            let v_r: CMatrix2 = staple.adjoint() / Complex::from(staple_coeef);
            let d_heat_bath_r = HeatBathDistribution::new(beta * staple_coeef).unwrap();
            let rand_m_r: CMatrix2 = self.rng.sample(d_heat_bath_r);
            rand_m_r * v_r
        }
        else{
            // if the determinant is 0 (or close to zero)
            su2::get_random_su2(&mut self.rng)
        }
    }
    
    #[inline]
    fn get_modif<const D: usize>(&mut self, state: &LatticeStateDefault<D>, link: &LatticeLinkCanonical<D>) -> na::Matrix3<Complex>
        where
        na::SVector<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let link_matrix = state.link_matrix().get_matrix(&link.into(), state.lattice()).unwrap();
        let a = get_staple(state.link_matrix(), state.lattice(), link);
        
        let r = su3::get_r(self.get_heat_bath_su2(su3::get_su2_r_unorm(link_matrix * a), state.beta()));
        let s = su3::get_s(self.get_heat_bath_su2(su3::get_su2_s_unorm(r * link_matrix * a), state.beta()));
        let t = su3::get_t(self.get_heat_bath_su2(su3::get_su2_t_unorm(s * r * link_matrix * a), state.beta()));
        
        t * s * r * link_matrix
    }
    
    #[inline]
    // TODO improve error handeling
    fn get_next_element_default<const D: usize>(&mut self, mut state: LatticeStateDefault<D>) -> LatticeStateDefault<D>
        where
        na::SVector<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = self.get_modif(&state, &link);
            *state.get_link_mut(&link).unwrap() = potential_modif;
        });
        state
    }
}

impl<Rng, const D: usize> MonteCarlo<LatticeStateDefault<D>, D> for HeatBathSweep<Rng>
    where Rng: rand::Rng,
    SVector<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Error = Never;
    
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault<D>) -> Result<LatticeStateDefault<D>, Self::Error> {
        Ok(self.get_next_element_default(state))
    }
}
