//! Overrelaxation method
//!
//! The goal of the overrelaxation is to move thought the phase space as much as possible but conserving the hamiltonian.
//! It can be used to improve the speed of thermalisation by vising more states.
//! Alone it can't advance the simulation as it preserved the hamiltonian. You need to use other method with this one.
//! You can look at [`super::HybridMethodVec`] and [`super::HybridMethodCouple`].
//!
//! In my limited experience [`OverrelaxationSweepReverse`] moves a bit more though the phase space than [`OverrelaxationSweepRotation`].
//! The difference is slight though.
//!
//! # Example
//! ```
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use lattice_qcd_rs::simulation::{HeatBathSweep, LatticeState, LatticeStateDefault, OverrelaxationSweepReverse};
//! use rand::SeedableRng;
//!
//! let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
//! let mut heat_bath = HeatBathSweep::new(rng);
//! let mut overrelax = OverrelaxationSweepReverse::default();
//!
//! let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 8_f64, 4)?; // 1_f64 : size, 8_f64: beta, 4 number of points.
//! for _ in 0..2 {
//!     state = state.monte_carlo_step(&mut heat_bath)?;
//!     state = state.monte_carlo_step(&mut overrelax)?;
//!     // operation to track the progress or the evolution
//! }
//! // operation at the end of the simulation
//! #     Ok(())
//! # }
//! ```

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        super::{error::Never, lattice::LatticeLinkCanonical, su3, Complex},
        state::{LatticeState, LatticeStateDefault},
    },
    staple, MonteCarlo,
};

/// Overrelaxation algorithm using rotation method.
///
/// Alone it can't advance the simulation as it preserved the hamiltonian.
/// You need to use other method with this one.
/// You can look at [`super::HybridMethodVec`] and [`super::HybridMethodCouple`].
///
/// The algorithm is described <https://arxiv.org/abs/hep-lat/0503041> in section 2.1 up to step 2 using `\hat X_{NN}`.
///
/// # Example
/// ```
/// # use std::error::Error;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// use lattice_qcd_rs::simulation::{HeatBathSweep, LatticeState, LatticeStateDefault, OverrelaxationSweepRotation};
/// use rand::SeedableRng;
///
/// let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
/// let mut heat_bath = HeatBathSweep::new(rng);
/// let mut overrelax = OverrelaxationSweepRotation::default();
///
/// let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 8_f64, 4)?; // 1_f64 : size, 8_f64: beta, 4 number of points.
/// for _ in 0..2 {
///     state = state.monte_carlo_step(&mut heat_bath)?;
///     state = state.monte_carlo_step(&mut overrelax)?;
///     // operation to track the progress or the evolution
/// }
/// // operation at the end of the simulation
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct OverrelaxationSweepRotation;

impl OverrelaxationSweepRotation {
    /// Create a new Self with an given RNG
    pub const fn new() -> Self {
        Self {}
    }

    #[inline]
    fn get_modif<const D: usize>(
        state: &LatticeStateDefault<D>,
        link: &LatticeLinkCanonical<D>,
    ) -> na::Matrix3<Complex> {
        let link_matrix = state
            .link_matrix()
            .matrix(&link.into(), state.lattice())
            .unwrap();
        let a = staple(state.link_matrix(), state.lattice(), link).adjoint();
        let svd = na::SVD::<Complex, na::U3, na::U3>::new(a, true, true);
        let rot = svd.u.unwrap() * svd.v_t.unwrap();
        rot * link_matrix.adjoint() * rot
    }

    #[inline]
    fn next_element_default<const D: usize>(
        mut state: LatticeStateDefault<D>,
    ) -> LatticeStateDefault<D> {
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = Self::get_modif(&state, &link);
            *state.link_mut(&link).unwrap() = potential_modif;
        });
        state
    }
}

impl std::fmt::Display for OverrelaxationSweepRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "overrelaxation method by rotation")
    }
}

impl Default for OverrelaxationSweepRotation {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> MonteCarlo<LatticeStateDefault<D>, D> for OverrelaxationSweepRotation {
    type Error = Never;

    #[inline]
    fn next_element(
        &mut self,
        state: LatticeStateDefault<D>,
    ) -> Result<LatticeStateDefault<D>, Self::Error> {
        Ok(Self::next_element_default(state))
    }
}

/// Overrelaxation algorithm using the reverse method.
///
/// Alone it can't advance the simulation as it preserved the hamiltonian.
/// You need to use other method with this one.
/// You can look at [`super::HybridMethodVec`] and [`super::HybridMethodCouple`].
///
/// The algorithm is described in <https://doi.org/10.1016/0370-2693(90)90032-2>.
///
/// # Example
/// see level module documentation [`super::overrelaxation`]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct OverrelaxationSweepReverse;

impl OverrelaxationSweepReverse {
    /// Create a new Self with an given RNG
    pub const fn new() -> Self {
        Self {}
    }

    #[inline]
    fn get_modif<const D: usize>(
        state: &LatticeStateDefault<D>,
        link: &LatticeLinkCanonical<D>,
    ) -> na::Matrix3<Complex> {
        let link_matrix = state
            .link_matrix()
            .matrix(&link.into(), state.lattice())
            .unwrap();
        let a = staple(state.link_matrix(), state.lattice(), link).adjoint();
        let svd = na::SVD::<Complex, na::U3, na::U3>::new(a, true, true);
        svd.u.unwrap()
            * su3::reverse(svd.u.unwrap().adjoint() * link_matrix * svd.v_t.unwrap().adjoint())
            * svd.v_t.unwrap()
    }

    #[inline]
    fn next_element_default<const D: usize>(
        mut state: LatticeStateDefault<D>,
    ) -> LatticeStateDefault<D> {
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = Self::get_modif(&state, &link);
            *state.link_mut(&link).unwrap() = potential_modif;
        });
        state
    }
}

impl Default for OverrelaxationSweepReverse {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for OverrelaxationSweepReverse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "overrelaxation method by reverse")
    }
}

impl<const D: usize> MonteCarlo<LatticeStateDefault<D>, D> for OverrelaxationSweepReverse {
    type Error = Never;

    #[inline]
    fn next_element(
        &mut self,
        state: LatticeStateDefault<D>,
    ) -> Result<LatticeStateDefault<D>, Self::Error> {
        Ok(Self::next_element_default(state))
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;

    use super::super::super::state::{LatticeState, LatticeStateDefault};
    use super::super::MonteCarlo;
    use super::*;

    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    fn test_same_energy<MC>(mc: &mut MC, rng: &mut impl rand::Rng)
    where
        MC: MonteCarlo<LatticeStateDefault<3>, 3>,
        MC::Error: core::fmt::Debug,
    {
        let state = LatticeStateDefault::<3>::new_determinist(1_f64, 1_f64, 4, rng).unwrap();
        let h = state.hamiltonian_links();
        let state2 = mc.next_element(state).unwrap();
        let h2 = state2.hamiltonian_links();
        println!("h1 {}, h2 {}", h, h2);
        // Relative assert : we need to multi by the mean value of h
        // TODO use crate approx ?
        assert!((h - h2).abs() < f64::EPSILON * 100_f64 * 4_f64.powi(3) * (h + h2) * 0.5_f64);
    }

    /// Here we test that OverrelaxationSweepReverse conserve the energy.
    #[test]
    fn same_energy_reverse() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let mut overrelax = OverrelaxationSweepReverse::new();
        for _ in 0_u32..10_u32 {
            test_same_energy(&mut overrelax, &mut rng);
        }
    }

    /// Here we test that OverrelaxationSweepRotation conserve the energy.
    #[test]
    fn same_energy_rotation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
        let mut overrelax = OverrelaxationSweepRotation::new();
        for _ in 0_u32..10_u32 {
            test_same_energy(&mut overrelax, &mut rng);
        }
    }
}
