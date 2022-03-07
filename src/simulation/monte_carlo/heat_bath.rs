//! Pseudo heat bath methods
//!
//! # Example
//! see [`HeatBathSweep`].

use na::ComplexField;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        super::{
            error::Never, lattice::LatticeLinkCanonical, statistics::HeatBathDistribution, su2,
            su3, CMatrix2, Complex,
        },
        state::{LatticeState, LatticeStateDefault},
    },
    staple, MonteCarlo,
};

/// Pseudo heat bath algorithm
///
/// # Example
/// ```
/// # use std::error::Error;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// use lattice_qcd_rs::simulation::{HeatBathSweep, LatticeState, LatticeStateDefault};
/// use rand::SeedableRng;
///
/// let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
/// let mut heat_bath = HeatBathSweep::new(rng);
///
/// let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 6_f64, 4)?;
/// for _ in 0..2 {
///     state = state.monte_carlo_step(&mut heat_bath)?;
///     // operation to track the progress or the evolution
/// }
/// // operation at the end of the simulation
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct HeatBathSweep<Rng: rand::Rng> {
    rng: Rng,
}

impl<Rng: rand::Rng> HeatBathSweep<Rng> {
    /// Create a new Self form a rng.
    pub fn new(rng: Rng) -> Self {
        Self { rng }
    }

    /// Absorbe self and return the RNG as owned. It essentialy deconstruct the structure.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }

    /// Get a mutable reference to the rng.
    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// Get a reference to the rng.
    pub fn rng(&self) -> &Rng {
        &self.rng
    }

    /// Apply te SU2 heat bath methode.
    #[inline]
    fn heat_bath_su2(&mut self, staple: CMatrix2, beta: f64) -> CMatrix2 {
        let staple_coeef = staple.determinant().real().sqrt();
        if staple_coeef.is_normal() {
            let v_r: CMatrix2 = staple.adjoint() / Complex::from(staple_coeef);
            let d_heat_bath_r = HeatBathDistribution::new(beta * staple_coeef).unwrap();
            let rand_m_r: CMatrix2 = self.rng.sample(d_heat_bath_r);
            rand_m_r * v_r
        }
        else {
            // just return a random matrix as all matrices
            // have the same selection probability
            su2::random_su2(&mut self.rng)
        }
    }

    /// Apply the pesudo heat bath methode and return the new link.
    #[inline]
    fn get_modif<const D: usize>(
        &mut self,
        state: &LatticeStateDefault<D>,
        link: &LatticeLinkCanonical<D>,
    ) -> na::Matrix3<Complex> {
        let link_matrix = state
            .link_matrix()
            .matrix(&link.into(), state.lattice())
            .unwrap();
        let a = staple(state.link_matrix(), state.lattice(), link);

        let r = su3::get_r(self.heat_bath_su2(su3::get_su2_r_unorm(link_matrix * a), state.beta()));
        let s =
            su3::get_s(self.heat_bath_su2(su3::get_su2_s_unorm(r * link_matrix * a), state.beta()));
        let t = su3::get_t(
            self.heat_bath_su2(su3::get_su2_t_unorm(s * r * link_matrix * a), state.beta()),
        );

        t * s * r * link_matrix
    }

    #[inline]
    // TODO improve error handeling
    fn next_element_default<const D: usize>(
        &mut self,
        mut state: LatticeStateDefault<D>,
    ) -> LatticeStateDefault<D> {
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = self.get_modif(&state, &link);
            *state.link_mut(&link).unwrap() = potential_modif;
        });
        state
    }
}

impl<Rng: rand::Rng> AsRef<Rng> for HeatBathSweep<Rng> {
    fn as_ref(&self) -> &Rng {
        self.rng()
    }
}

impl<Rng: rand::Rng> AsMut<Rng> for HeatBathSweep<Rng> {
    fn as_mut(&mut self) -> &mut Rng {
        self.rng_mut()
    }
}

impl<Rng: rand::Rng + Default> Default for HeatBathSweep<Rng> {
    fn default() -> Self {
        Self::new(Rng::default())
    }
}

impl<Rng, const D: usize> MonteCarlo<LatticeStateDefault<D>, D> for HeatBathSweep<Rng>
where
    Rng: rand::Rng,
{
    type Error = Never;

    #[inline]
    fn next_element(
        &mut self,
        state: LatticeStateDefault<D>,
    ) -> Result<LatticeStateDefault<D>, Self::Error> {
        Ok(self.next_element_default(state))
    }
}
