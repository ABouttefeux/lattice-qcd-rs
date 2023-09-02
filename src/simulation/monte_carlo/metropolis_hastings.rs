//! Metropolis Hastings method
//!
//! I recommend not using method in this module, but they may have niche usage.
//! look at [`super::metropolis_hastings_sweep`] for a more common algorithm.
//!
//! # Example
//! ```rust
//! use lattice_qcd_rs::{
//!     error::ImplementationError,
//!     simulation::monte_carlo::MetropolisHastingsDeltaDiagnostic,
//!     simulation::state::{LatticeState, LatticeStateDefault},
//!     ComplexField,
//! };
//!
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! let mut rng = rand::thread_rng();
//!
//! let size = 1_000_f64;
//! let number_of_pts = 4;
//! let beta = 2_f64;
//! let mut simulation =
//!     LatticeStateDefault::<4>::new_determinist(size, beta, number_of_pts, &mut rng)?;
//!
//! let spread_parameter = 1E-5_f64;
//! let mut mc = MetropolisHastingsDeltaDiagnostic::new(spread_parameter, rng)
//!     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
//!
//! let number_of_sims = 100;
//! for _ in 0..number_of_sims / 10 {
//!     for _ in 0..10 {
//!         simulation = simulation.monte_carlo_step(&mut mc)?;
//!     }
//!     simulation.normalize_link_matrices(); // we renormalize all matrices back to SU(3);
//! }
//! let average = simulation
//!     .average_trace_plaquette()
//!     .ok_or(ImplementationError::OptionWithUnexpectedNone)?
//!     .real();
//! # Ok(())
//! # }
//! ```

use std::fmt::{self, Display};

use rand_distr::{Bernoulli, Distribution, Uniform};
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        super::{
            error::Never,
            field::LinkMatrix,
            lattice::{
                Direction, LatticeCyclic, LatticeElementToIndex, LatticeLink, LatticeLinkCanonical,
                LatticePoint,
            },
            su3, Complex, Real,
        },
        state::{LatticeState, LatticeStateDefault, LatticeStateNew},
    },
    delta_s_old_new_cmp, MonteCarlo, MonteCarloDefault,
};

/// Metropolis Hastings algorithm. Very slow, use [`MetropolisHastingsDeltaDiagnostic`]
/// instead when applicable.
///
/// This a very general method that can manage every [`LatticeState`] but the tread off
/// is that it is much slower than
/// a dedicated algorithm knowing the from of the hamiltonian. If you want to use your own
/// hamiltonian I advice to implement
/// you own method too.
///
/// Note that this method does not do a sweep but change random link matrix,
/// for a sweep there is [`super::MetropolisHastingsSweep`].
///
/// # Example
/// See the example of [`super::McWrapper`]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastings {
    number_of_update: usize,
    spread: Real,
}

impl MetropolisHastings {
    /// `spread` should be between 0 and 1 both not included and `number_of_update` should be greater
    /// than 0. `0.1_f64` is a good choice for this parameter.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spread factor for the random matrix change
    /// ( used in [`su3::random_su3_close_to_unity`]).
    #[inline]
    #[must_use]
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || !(spread > 0_f64 && spread < 1_f64) {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
        })
    }

    getter_copy!(
        /// Get the number of attempted updates per steps.
        #[inline]
        #[must_use]
        pub const number_of_update() -> usize
    );

    getter_copy!(
        /// Get the spread parameter.
        #[inline]
        #[must_use]
        pub const spread() -> Real
    );
}

impl Default for MetropolisHastings {
    #[inline]
    fn default() -> Self {
        Self::new(1, 0.1_f64).expect("always exist")
    }
}

impl Display for MetropolisHastings {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Metropolis-Hastings method with {} update and spread {}",
            self.number_of_update(),
            self.spread()
        )
    }
}

impl<State, const D: usize> MonteCarloDefault<State, D> for MetropolisHastings
where
    State: LatticeState<D> + LatticeStateNew<D>,
{
    type Error = State::Error;

    #[inline]
    fn potential_next_element<Rng>(
        &mut self,
        state: &State,
        rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
        let d = Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::random_su3_close_to_unity(self.spread, rng);
        });
        State::new(
            state.lattice().clone(),
            state.beta(),
            LinkMatrix::new(link_matrix),
        )
    }
}

/// Metropolis Hastings algorithm with diagnostics. Very slow, use [`MetropolisHastingsDeltaDiagnostic`] instead.
///
/// Similar to [`MetropolisHastingsDiagnostic`] but with diagnostic information.
///
/// Note that this method does not do a sweep but change random link matrix,
/// for a sweep there is [`super::MetropolisHastingsSweep`].
///
/// # Example
/// see example of [`super::McWrapper`]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastingsDiagnostic {
    number_of_update: usize,
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
}

impl MetropolisHastingsDiagnostic {
    /// `spread` should be between 0 and 1 both not included and `number_of_update` should be greater
    /// than 0. `0.1_f64` is a good choice for this parameter.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spread factor for the random matrix change
    /// ( used in [`su3::random_su3_close_to_unity`]).
    #[inline]
    #[must_use]
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
            has_replace_last: false,
            prob_replace_last: 0_f64,
        })
    }

    /// Get the last probably of acceptance of the random change.
    #[inline]
    #[must_use]
    pub const fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }

    /// Get if last step has accepted the replacement.
    #[inline]
    #[must_use]
    pub const fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }

    getter_copy!(
        /// Get the number of updates per steps.
        #[inline]
        #[must_use]
        pub const number_of_update() -> usize
    );

    getter_copy!(
        /// Get the spread parameter.
        #[inline]
        #[must_use]
        pub const spread() -> Real
    );
}

impl Default for MetropolisHastingsDiagnostic {
    #[inline]
    fn default() -> Self {
        Self::new(1, 0.1_f64).expect("always exist")
    }
}

impl Display for MetropolisHastingsDiagnostic {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Metropolis-Hastings method with {} update and spread {}, with diagnostics: has accepted last step {}, probability of acceptance of last step {}",
            self.number_of_update(),
            self.spread(),
            self.has_replace_last(),
            self.prob_replace_last()
        )
    }
}

impl<State, const D: usize> MonteCarloDefault<State, D> for MetropolisHastingsDiagnostic
where
    State: LatticeState<D> + LatticeStateNew<D>,
{
    type Error = State::Error;

    #[inline]
    fn potential_next_element<Rng>(
        &mut self,
        state: &State,
        rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
        let d = Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::random_su3_close_to_unity(self.spread, rng);
        });
        State::new(
            state.lattice().clone(),
            state.beta(),
            LinkMatrix::new(link_matrix),
        )
    }

    #[allow(clippy::unwrap_in_result)] // expect does not panic because of the clamp
    #[inline]
    fn next_element_default<Rng>(
        &mut self,
        state: State,
        rng: &mut Rng,
    ) -> Result<State, Self::Error>
    where
        Rng: rand::Rng + ?Sized,
    {
        let potential_next = self.potential_next_element(&state, rng)?;
        let proba = Self::probability_of_replacement(&state, &potential_next).clamp(0_f64, 1_f64); // cspell: ignore proba
        self.prob_replace_last = proba;
        let d = Bernoulli::new(proba).expect("always exist because of the clamp");
        if d.sample(rng) {
            self.has_replace_last = true;
            Ok(potential_next)
        } else {
            self.has_replace_last = false;
            Ok(state)
        }
    }
}

/// Metropolis Hastings algorithm with diagnostics.
///
/// Note that this method does not do a sweep but change random link matrix,
/// for a sweep there is [`super::MetropolisHastingsSweep`].
///
/// # Example
/// see example of [`super`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastingsDeltaDiagnostic<Rng: rand::Rng> {
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
    rng: Rng,
}

impl<Rng: rand::Rng> MetropolisHastingsDeltaDiagnostic<Rng> {
    getter_copy!(
        /// Get the last probably of acceptance of the random change.
        #[inline]
        #[must_use]
        pub const,
        prob_replace_last,
        Real
    );

    getter_copy!(
        /// Get if last step has accepted the replacement.
        #[inline]
        #[must_use]
        pub const,
        has_replace_last,
        bool
    );

    getter!(
        /// Get a ref to the rng.
        #[inline]
        #[must_use]
        pub const,
        rng,
        Rng
    );

    getter_copy!(
        /// Get the spread parameter.
        #[inline]
        #[must_use]
        pub const spread() -> Real
    );

    /// Get a mutable reference to the rng.
    #[inline]
    #[must_use]
    pub fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// `spread` should be between 0 and 1 both not included and `number_of_update` should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spread factor for the random matrix change
    /// ( used in [`su3::random_su3_close_to_unity`]).
    #[inline]
    #[must_use]
    pub fn new(spread: Real, rng: Rng) -> Option<Self> {
        if spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            spread,
            has_replace_last: false,
            prob_replace_last: 0_f64,
            rng,
        })
    }

    /// Absorbs self and return the RNG as owned. It essentially deconstruct the structure.
    #[inline]
    #[must_use]
    pub fn rng_owned(self) -> Rng {
        self.rng
    }

    #[inline]
    #[must_use]
    fn delta_s<const D: usize>(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclic<D>,
        link: &LatticeLinkCanonical<D>,
        new_link: &nalgebra::Matrix3<Complex>,
        beta: Real,
    ) -> Real {
        let old_matrix = link_matrix
            .matrix(&LatticeLink::from(*link), lattice)
            .expect("matrix not found");
        delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }

    // cspell: ignore modif
    #[inline]
    #[must_use]
    fn potential_modif<const D: usize>(
        &mut self,
        state: &LatticeStateDefault<D>,
    ) -> (LatticeLinkCanonical<D>, nalgebra::Matrix3<Complex>) {
        let d_p = Uniform::new(0, state.lattice().dim());
        let d_d = Uniform::new(0, LatticeCyclic::<D>::dim_st());

        let point = LatticePoint::from_fn(|_| d_p.sample(&mut self.rng));
        let direction = Direction::positive_directions()[d_d.sample(&mut self.rng)];
        let link = LatticeLinkCanonical::new(point, direction).expect("unreachable");
        let index = link.to_index(state.lattice());

        let old_link_m = state.link_matrix()[index];
        let rand_m =
            su3::orthonormalize_matrix(&su3::random_su3_close_to_unity(self.spread, &mut self.rng));
        let new_link = rand_m * old_link_m;
        (link, new_link)
    }

    #[inline]
    #[must_use]
    fn next_element_default<const D: usize>(
        &mut self,
        mut state: LatticeStateDefault<D>,
    ) -> LatticeStateDefault<D> {
        let (link, matrix) = self.potential_modif(&state);
        let delta_s = Self::delta_s(
            state.link_matrix(),
            state.lattice(),
            &link,
            &matrix,
            state.beta(),
        );
        let proba = (-delta_s).exp().clamp(0_f64, 1_f64);
        self.prob_replace_last = proba;
        let d = Bernoulli::new(proba).expect("always exist");
        if d.sample(&mut self.rng) {
            self.has_replace_last = true;
            *state.link_mut(&link).expect("link not found") = matrix;
        } else {
            self.has_replace_last = false;
        }
        state
    }
}

impl<Rng: rand::Rng + Default> Default for MetropolisHastingsDeltaDiagnostic<Rng> {
    #[inline]
    fn default() -> Self {
        Self::new(0.1_f64, Rng::default()).expect("unreachable")
    }
}

impl<Rng: rand::Rng + Display> Display for MetropolisHastingsDeltaDiagnostic<Rng> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Metropolis-Hastings delta method with rng {} and spread {}, with diagnostics: has accepted last step {}, probability of acceptance of last step {}",
            self.rng(),
            self.spread(),
            self.has_replace_last(),
            self.prob_replace_last()
        )
    }
}

impl<Rng: rand::Rng> AsRef<Rng> for MetropolisHastingsDeltaDiagnostic<Rng> {
    #[inline]
    fn as_ref(&self) -> &Rng {
        self.rng()
    }
}

impl<Rng: rand::Rng> AsMut<Rng> for MetropolisHastingsDeltaDiagnostic<Rng> {
    #[inline]
    fn as_mut(&mut self) -> &mut Rng {
        self.rng_mut()
    }
}

impl<Rng, const D: usize> MonteCarlo<LatticeStateDefault<D>, D>
    for MetropolisHastingsDeltaDiagnostic<Rng>
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

    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{error::ImplementationError, simulation::state::*};

    const SEED: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    fn test_mh_delta() -> Result<(), Box<dyn Error>> {
        let mut rng = StdRng::seed_from_u64(SEED);

        let size = 1_000_f64;
        let number_of_pts = 4;
        let beta = 2_f64;
        let mut simulation =
            LatticeStateDefault::<4>::new_determinist(size, beta, number_of_pts, &mut rng)?;

        let mut mcd = MetropolisHastingsDeltaDiagnostic::new(0.01_f64, rng)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        for _ in 0_u32..10_u32 {
            let mut simulation2 = simulation.clone();
            let (link, matrix) = mcd.potential_modif(&simulation);
            *simulation2
                .link_mut(&link)
                .ok_or(ImplementationError::OptionWithUnexpectedNone)? = matrix;
            let ds = MetropolisHastingsDeltaDiagnostic::<StdRng>::delta_s(
                simulation.link_matrix(),
                simulation.lattice(),
                &link,
                &matrix,
                simulation.beta(),
            );
            println!(
                "ds {}, dh {}",
                ds,
                -simulation.hamiltonian_links() + simulation2.hamiltonian_links()
            );
            let prob_of_replacement = (simulation.hamiltonian_links()
                - simulation2.hamiltonian_links())
            .exp()
            .clamp(0_f64, 1_f64);
            assert!(((-ds).exp().clamp(0_f64, 1_f64) - prob_of_replacement).abs() < 1E-8_f64);
            simulation = simulation2;
        }
        Ok(())
    }
    #[test]
    fn methods_common_traits() -> Result<(), ImplementationError> {
        assert_eq!(
            Some(MetropolisHastings::default()),
            MetropolisHastings::new(1, 0.1_f64)
        );
        assert_eq!(
            Some(MetropolisHastingsDiagnostic::default()),
            MetropolisHastingsDiagnostic::new(1, 0.1_f64)
        );

        let rng = StdRng::seed_from_u64(SEED);
        assert!(MetropolisHastingsDeltaDiagnostic::new(0_f64, rng.clone()).is_none());
        assert!(MetropolisHastings::new(0, 0.1_f64).is_none());
        assert!(MetropolisHastingsDiagnostic::new(1, 0_f64).is_none());

        assert_eq!(
            MetropolisHastings::new(2, 0.2_f64)
                .ok_or(ImplementationError::OptionWithUnexpectedNone)?
                .to_string(),
            "Metropolis-Hastings method with 2 update and spread 0.2"
        );
        assert_eq!(
            MetropolisHastingsDiagnostic::new(2, 0.2_f64).ok_or(ImplementationError::OptionWithUnexpectedNone)?.to_string(),
            "Metropolis-Hastings method with 2 update and spread 0.2, with diagnostics: has accepted last step false, probability of acceptance of last step 0"
        );
        let mut mhdd = MetropolisHastingsDeltaDiagnostic::new(0.1_f64, rng)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?; // cspell: ignore mhdd
        let _: &StdRng = mhdd.as_ref();
        let _: &mut StdRng = mhdd.as_mut();
        Ok(())
    }
}
