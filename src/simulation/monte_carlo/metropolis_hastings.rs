//! Metropolis Hastings methode

use rand_distr::Distribution;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    super::{
        super::{
            error::Never,
            field::LinkMatrix,
            lattice::{
                Direction, LatticeCyclique, LatticeElementToIndex, LatticeLink,
                LatticeLinkCanonical, LatticePoint,
            },
            su3, Complex, Real,
        },
        state::{LatticeState, LatticeStateDefault, LatticeStateNew},
    },
    get_delta_s_old_new_cmp, MonteCarlo, MonteCarloDefault,
};

/// Metropolis Hastings algorithm. Very slow, use [`MetropolisHastingsDeltaDiagnostic`] instead.
///
/// Note that this methode does not do a sweep but change random link matrix,
/// for a sweep there is [`super::MetropolisHastingsSweep`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastings {
    number_of_update: usize,
    spread: Real,
}

impl MetropolisHastings {
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
    pub fn new(number_of_update: usize, spread: Real) -> Option<Self> {
        if number_of_update == 0 || spread <= 0_f64 || spread >= 1_f64 {
            return None;
        }
        Some(Self {
            number_of_update,
            spread,
        })
    }
}

impl<State, const D: usize> MonteCarloDefault<State, D> for MetropolisHastings
where
    State: LatticeState<D> + LatticeStateNew<D>,
{
    type Error = State::Error;

    fn get_potential_next_element(
        &mut self,
        state: &State,
        rng: &mut impl rand::Rng,
    ) -> Result<State, Self::Error> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
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
/// Note that this methode does not do a sweep but change random link matrix,
/// for a sweep there is [`super::MetropolisHastingsSweep`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct MetropolisHastingsDiagnostic {
    number_of_update: usize,
    spread: Real,
    has_replace_last: bool,
    prob_replace_last: Real,
}

impl MetropolisHastingsDiagnostic {
    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
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
    pub const fn prob_replace_last(&self) -> Real {
        self.prob_replace_last
    }

    /// Get if last step has accepted the replacement.
    pub const fn has_replace_last(&self) -> bool {
        self.has_replace_last
    }
}

impl<State, const D: usize> MonteCarloDefault<State, D> for MetropolisHastingsDiagnostic
where
    State: LatticeState<D> + LatticeStateNew<D>,
{
    type Error = State::Error;

    fn get_potential_next_element(
        &mut self,
        state: &State,
        rng: &mut impl rand::Rng,
    ) -> Result<State, Self::Error> {
        let d = rand::distributions::Uniform::new(0, state.link_matrix().len());
        let mut link_matrix = state.link_matrix().data().clone();
        (0..self.number_of_update).for_each(|_| {
            let pos = d.sample(rng);
            link_matrix[pos] *= su3::get_random_su3_close_to_unity(self.spread, rng);
        });
        State::new(
            state.lattice().clone(),
            state.beta(),
            LinkMatrix::new(link_matrix),
        )
    }

    fn get_next_element_default(
        &mut self,
        state: State,
        rng: &mut impl rand::Rng,
    ) -> Result<State, Self::Error> {
        let potential_next = self.get_potential_next_element(&state, rng)?;
        let proba = Self::get_probability_of_replacement(&state, &potential_next)
            .min(1_f64)
            .max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(rng) {
            self.has_replace_last = true;
            Ok(potential_next)
        }
        else {
            self.has_replace_last = false;
            Ok(state)
        }
    }
}

/// Metropolis Hastings algorithm with diagnostics.
///
/// Note that this methode does not do a sweep but change random link matrix,
/// for a sweep there is [`super::MetropolisHastingsSweep`].
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
        prob_replace_last,
        Real
    );

    getter_copy!(
        /// Get if last step has accepted the replacement.
        has_replace_last,
        bool
    );

    /// `spread` should be between 0 and 1 both not included and number_of_update should be greater
    /// than 0.
    ///
    /// `number_of_update` is the number of times a link matrix is randomly changed.
    /// `spread` is the spead factor for the random matrix change
    /// ( used in [`su3::get_random_su3_close_to_unity`]).
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

    /// Absorbe self and return the RNG as owned. It essentialy deconstruct the structure.
    pub fn rng_owned(self) -> Rng {
        self.rng
    }

    #[inline]
    fn get_delta_s<const D: usize>(
        link_matrix: &LinkMatrix,
        lattice: &LatticeCyclique<D>,
        link: &LatticeLinkCanonical<D>,
        new_link: &na::Matrix3<Complex>,
        beta: Real,
    ) -> Real {
        let old_matrix = link_matrix
            .get_matrix(&LatticeLink::from(*link), lattice)
            .unwrap();
        get_delta_s_old_new_cmp(link_matrix, lattice, link, new_link, beta, &old_matrix)
    }

    #[inline]
    fn get_potential_modif<const D: usize>(
        &mut self,
        state: &LatticeStateDefault<D>,
    ) -> (LatticeLinkCanonical<D>, na::Matrix3<Complex>) {
        let d_p = rand::distributions::Uniform::new(0, state.lattice().dim());
        let d_d = rand::distributions::Uniform::new(0, LatticeCyclique::<D>::dim_st());

        let point = LatticePoint::from_fn(|_| d_p.sample(&mut self.rng));
        let direction = Direction::positive_directions()[d_d.sample(&mut self.rng)];
        let link = LatticeLinkCanonical::new(point, direction).unwrap();
        let index = link.to_index(state.lattice());

        let old_link_m = state.link_matrix()[index];
        let rand_m = su3::orthonormalize_matrix(&su3::get_random_su3_close_to_unity(
            self.spread,
            &mut self.rng,
        ));
        let new_link = rand_m * old_link_m;
        (link, new_link)
    }

    #[inline]
    fn get_next_element_default<const D: usize>(
        &mut self,
        mut state: LatticeStateDefault<D>,
    ) -> LatticeStateDefault<D> {
        let (link, matrix) = self.get_potential_modif(&state);
        let delta_s = Self::get_delta_s(
            state.link_matrix(),
            state.lattice(),
            &link,
            &matrix,
            state.beta(),
        );
        let proba = (-delta_s).exp().min(1_f64).max(0_f64);
        self.prob_replace_last = proba;
        let d = rand::distributions::Bernoulli::new(proba).unwrap();
        if d.sample(&mut self.rng) {
            self.has_replace_last = true;
            *state.get_link_mut(&link).unwrap() = matrix;
            state
        }
        else {
            self.has_replace_last = false;
            state
        }
    }
}

impl<Rng, const D: usize> MonteCarlo<LatticeStateDefault<D>, D>
    for MetropolisHastingsDeltaDiagnostic<Rng>
where
    Rng: rand::Rng,
{
    type Error = Never;

    #[inline]
    fn get_next_element(
        &mut self,
        state: LatticeStateDefault<D>,
    ) -> Result<LatticeStateDefault<D>, Self::Error> {
        Ok(self.get_next_element_default(state))
    }
}

#[cfg(test)]
#[test]
fn test_mh_delta() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x45_78_93_f4_4a_b0_67_f0);

    let size = 1_000_f64;
    let number_of_pts = 4;
    let beta = 2_f64;
    let mut simulation =
        LatticeStateDefault::<4>::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();

    let mut mcd = MetropolisHastingsDeltaDiagnostic::new(0.01, rng).unwrap();
    for _ in 0..10 {
        let mut simulation2 = simulation.clone();
        let (link, matrix) = mcd.get_potential_modif(&simulation);
        *simulation2.get_link_mut(&link).unwrap() = matrix;
        let ds = MetropolisHastingsDeltaDiagnostic::<rand::rngs::StdRng>::get_delta_s(
            simulation.link_matrix(),
            simulation.lattice(),
            &link,
            &matrix,
            simulation.beta(),
        );
        println!(
            "ds {}, dh {}",
            ds,
            -simulation.get_hamiltonian_links() + simulation2.get_hamiltonian_links()
        );
        let prob_of_replacement = (simulation.get_hamiltonian_links()
            - simulation2.get_hamiltonian_links())
        .exp()
        .min(1_f64)
        .max(0_f64);
        assert!(((-ds).exp().min(1_f64).max(0_f64) - prob_of_replacement).abs() < 1E-8_f64);
        simulation = simulation2;
    }
}
