//! Metropolis Hastings method
//!
//! # Example
//! see [`MetropolisHastingsSweep`]

use rand_distr::Distribution;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        super::{
            error::Never,
            field::LinkMatrix,
            lattice::{LatticeCyclic, LatticeElementToIndex, LatticeLink, LatticeLinkCanonical},
            su3, Complex, Real,
        },
        state::{LatticeState, LatticeStateDefault},
    },
    delta_s_old_new_cmp, MonteCarlo,
};

/// Metropolis Hastings method by doing a pass on all points
///
/// # Example
/// ```
/// # use std::error::Error;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// use lattice_qcd_rs::error::ImplementationError;
/// use lattice_qcd_rs::simulation::{LatticeState, LatticeStateDefault, MetropolisHastingsSweep};
/// use rand::SeedableRng;
///
/// let rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
/// let mut mh = MetropolisHastingsSweep::new(1, 0.1_f64, rng)
///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// // Realistically you want more steps than 10
///
/// let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 6_f64, 4)?;
/// for _ in 0..10 {
///     state = state.monte_carlo_step(&mut mh)?;
///     println!(
///         "mean probability of acceptance during last step {} and replaced {} links",
///         mh.prob_replace_mean(),
///         mh.number_replace_last()
///     );
///     // operation to track the progress or the evolution
/// }
/// // operation at the end of the simulation
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastingsSweep<Rng: rand::Rng> {
    number_of_update: usize,
    spread: Real,
    number_replace_last: usize,
    prob_replace_mean: Real,
    rng: Rng,
}

impl<Rng: rand::Rng> MetropolisHastingsSweep<Rng> {
    getter!(
        /// Get a ref to the rng.
        pub const,
        rng,
        Rng
    );

    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spread factor for the random matrix change
    /// ( used in [`su3::random_su3_close_to_unity`]).
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
    pub const fn prob_replace_mean(&self) -> Real {
        self.prob_replace_mean
    }

    /// Number of accepted change during last sweep
    pub const fn number_replace_last(&self) -> usize {
        self.number_replace_last
    }

    /// Get the last probably of acceptance of the random change.
    #[allow(clippy::missing_const_for_fn)] // false positive
    pub fn rng_owned(self) -> Rng {
        self.rng
    }

    /// Get a mutable reference to the rng.
    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    #[inline]
    fn delta_s<const D: usize>(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclic<D>,
        link: &LatticeLinkCanonical<D>,
        new_link: &nalgebra::Matrix3<Complex>,
        beta: Real,
    ) -> Real {
        let old_matrix = link_matrix
            .matrix(&LatticeLink::from(*link), lattice)
            .unwrap();
        delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }

    #[inline]
    fn potential_modif<const D: usize>(
        &mut self,
        state: &LatticeStateDefault<D>,
        link: &LatticeLinkCanonical<D>,
    ) -> nalgebra::Matrix3<Complex> {
        let index = link.to_index(state.lattice());
        let old_link_m = state.link_matrix()[index];
        let mut new_link = old_link_m;
        for _ in 0..self.number_of_update {
            let rand_m = su3::orthonormalize_matrix(&su3::random_su3_close_to_unity(
                self.spread,
                &mut self.rng,
            ));
            new_link = rand_m * new_link;
        }

        new_link
    }

    #[inline]
    fn next_element_default<const D: usize>(
        &mut self,
        mut state: LatticeStateDefault<D>,
    ) -> LatticeStateDefault<D> {
        self.prob_replace_mean = 0_f64;
        self.number_replace_last = 0;
        let lattice = state.lattice().clone();
        lattice.get_links().for_each(|link| {
            let potential_modif = self.potential_modif(&state, &link);
            let proba = (-Self::delta_s(
                state.link_matrix(),
                state.lattice(),
                &link,
                &potential_modif,
                state.beta(),
            ))
            .exp()
            .min(1_f64)
            .max(0_f64);
            self.prob_replace_mean += proba;
            let d = rand::distributions::Bernoulli::new(proba).unwrap();
            if d.sample(&mut self.rng) {
                self.number_replace_last += 1;
                *state.link_mut(&link).unwrap() = potential_modif;
            }
        });
        self.prob_replace_mean /= lattice.number_of_canonical_links_space() as f64;
        state
    }
}

impl<Rng: rand::Rng> AsRef<Rng> for MetropolisHastingsSweep<Rng> {
    fn as_ref(&self) -> &Rng {
        self.rng()
    }
}

impl<Rng: rand::Rng> AsMut<Rng> for MetropolisHastingsSweep<Rng> {
    fn as_mut(&mut self) -> &mut Rng {
        self.rng_mut()
    }
}

impl<Rng, const D: usize> MonteCarlo<LatticeStateDefault<D>, D> for MetropolisHastingsSweep<Rng>
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

#[cfg(test)]
mod test {
    use std::error::Error;

    use rand::SeedableRng;

    use super::*;
    use crate::error::ImplementationError;

    #[test]
    fn as_ref_as_mut() -> Result<(), Box<dyn Error>> {
        let rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut mh = MetropolisHastingsSweep::new(1, 0.1_f64, rng.clone())
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq!(&rng, mh.as_ref());

        let _: &mut rand::rngs::StdRng = mh.as_mut();

        Ok(())
    }
}
