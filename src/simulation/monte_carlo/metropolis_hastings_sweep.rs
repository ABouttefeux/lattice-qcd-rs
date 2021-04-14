
//! Metropolis Hastings methode

use super::{
    MonteCarlo,
    get_delta_s_old_new_cmp,
    super::{
        super::{
            Real,
            Complex,
            field::{
                LinkMatrix,
            },
            su3,
            lattice::{
                LatticeLinkCanonical,
                Direction,
                LatticeElementToIndex,
                LatticeLink,
                LatticeCyclique,
                DirectionList,
            },
            error::Never,
        },
        state::{
            LatticeState,
            LatticeStateDefault,
        },
    },
};
use rand_distr::Distribution;
use na::{
    SVector,
};
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};

/// Metropolis Hastings methode by doing a pass on all points
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastingsSweep<Rng>
    where Rng: rand::Rng,
{
    number_of_update: usize,
    spread: Real,
    number_replace_last: usize,
    prob_replace_mean: Real,
    rng: Rng
}

impl<Rng> MetropolisHastingsSweep<Rng>
    where Rng: rand::Rng,
{
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real, rng: Rng) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            number_replace_last: 0,
            prob_replace_mean: 0_f64,
            rng,
        })
    }
    
    /// Get the mean of last probably of acceptance of the random change.
    pub fn prob_replace_mean(&self) -> Real {
        self.prob_replace_mean
    }
    
    /// Number of accepted chnage during last sweep
    pub fn number_replace_last(&self) -> usize {
        self.number_replace_last
    }
    
    /// Get the last probably of acceptance of the random change.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }
    
    #[inline]
    fn get_delta_s<const D: usize>(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
        link: &LatticeLinkCanonical<D>,
        new_link: &na::Matrix3<Complex>,
        beta : Real,
    ) -> Real
        where na::SVector<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let old_matrix = link_matrix.get_matrix(&LatticeLink::from(*link), lattice).unwrap();
        get_delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }
    
    #[inline]
    fn get_potential_modif<const D: usize>(&mut self, state: &LatticeStateDefault<D>, link: &LatticeLinkCanonical<D>) -> na::Matrix3<Complex>
        where
        na::SVector<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        let index = link.to_index(state.lattice());
        let old_link_m = state.link_matrix()[index];
        let mut new_link = old_link_m;
        for _ in 0..self.number_of_update {
            let rand_m = su3::orthonormalize_matrix(&su3::get_random_su3_close_to_unity(self.spread, &mut self.rng));
            new_link = rand_m * new_link;
        };
        
        new_link
    }
    
    #[inline]
    fn get_next_element_default<const D: usize>(&mut self, mut state: LatticeStateDefault<D>) -> LatticeStateDefault<D>
        where
        na::SVector<usize, D>: Copy + Send + Sync,
        Direction<D>: DirectionList,
    {
        self.prob_replace_mean = 0_f64;
        self.number_replace_last += 0;
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = self.get_potential_modif(&state, &link);
            let proba = (-Self::get_delta_s(state.link_matrix(), state.lattice(), &link, &potential_modif, state.beta())).exp().min(1_f64).max(0_f64);
            self.prob_replace_mean += proba;
            let d = rand::distributions::Bernoulli::new(proba).unwrap();
            if d.sample(&mut self.rng) {
                self.number_replace_last += 1;
                *state.get_link_mut(&link).unwrap() = potential_modif;
            }
        });
        self.prob_replace_mean /= lattice.get_number_of_canonical_links_space() as f64;
        state
    }
}

impl<Rng, const D: usize> MonteCarlo<LatticeStateDefault<D>, D> for MetropolisHastingsSweep<Rng>
    where Rng: rand::Rng,
    SVector<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Error = Never;
    #[inline]
    fn get_next_element(&mut self, state: LatticeStateDefault<D>) -> Result<LatticeStateDefault<D>, Self::Error>{
        Ok(self.get_next_element_default(state))
    }
}
