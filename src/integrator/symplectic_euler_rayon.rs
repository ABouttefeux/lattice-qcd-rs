//! Basic symplectic Euler integrator using [`Rayon`](https://docs.rs/rayon/1.5.1/rayon/).

use std::fmt::{self, Display};

use nalgebra::SVector;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{integrate_efield, integrate_link, CMatrix3, SymplecticIntegrator};
use crate::{
    field::{EField, LinkMatrix, Su3Adjoint},
    lattice::LatticeCyclic,
    simulation::{
        LatticeState, LatticeStateWithEField, LatticeStateWithEFieldNew, SimulationStateLeap,
        SimulationStateSynchronous,
    },
    thread::run_pool_parallel_rayon,
    Real,
};

/// Basic symplectic Euler integrator using Rayon.
///
/// It is slightly faster than [`super::SymplecticEuler`] but use slightly more memory.
/// # Example
/// ```
/// # use std::error::Error;
/// #
/// # fn main() -> Result<(), Box<dyn Error>> {
/// use lattice_qcd_rs::integrator::{SymplecticEulerRayon, SymplecticIntegrator};
/// use lattice_qcd_rs::simulation::{
///     LatticeStateDefault, LatticeStateEFSyncDefault, LatticeStateWithEField,
/// };
/// use rand::SeedableRng;
///
/// let mut rng = rand::rngs::StdRng::seed_from_u64(0); // change with your seed
/// let state1 = LatticeStateEFSyncDefault::new_random_e_state(
///     LatticeStateDefault::<3>::new_determinist(1_f64, 2_f64, 4, &mut rng)?,
///     &mut rng,
/// );
/// let integrator = SymplecticEulerRayon::default();
/// let state2 = integrator.integrate_symplectic(&state1, 0.000_001_f64)?;
/// let h = state1.hamiltonian_total();
/// let h2 = state2.hamiltonian_total();
/// println!("The error on the Hamiltonian is {}", h - h2);
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SymplecticEulerRayon;

impl SymplecticEulerRayon {
    /// Create a new [`SymplecticEulerRayon`]
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Get all the integrated links
    /// # Panics
    /// panics if the lattice has fewer link than `link_matrix` has or it has fewer point than `e_field` has.
    /// In debug panic if the lattice has not the same number link as `link_matrix`
    /// or not the same number of point as `e_field`.
    #[must_use]
    #[inline]
    fn link_matrix_integrate<State, const D: usize>(
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
        delta_t: Real,
    ) -> Vec<CMatrix3>
    where
        State: LatticeStateWithEField<D>,
    {
        // TODO improve perf
        run_pool_parallel_rayon(
            lattice.get_links(),
            &(link_matrix, e_field, lattice),
            |link, (link_matrix, e_field, lattice)| {
                integrate_link::<State, D>(link, link_matrix, e_field, lattice, delta_t)
            },
        )
    }

    /// Get all the integrated e field
    /// # Panics
    /// panics if the lattice has fewer link than `link_matrix` has or it has fewer point than `e_field` has.
    /// In debug panic if the lattice has not the same number link as `link_matrix`
    /// or not the same number of point as `e_field`.
    #[must_use]
    #[inline]
    fn e_field_integrate<State, const D: usize>(
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
        delta_t: Real,
    ) -> Vec<SVector<Su3Adjoint, D>>
    where
        State: LatticeStateWithEField<D>,
    {
        // TODO improve perf
        run_pool_parallel_rayon(
            lattice.get_points(),
            &(link_matrix, e_field, lattice),
            |point, (link_matrix, e_field, lattice)| {
                integrate_efield::<State, D>(point, link_matrix, e_field, lattice, delta_t)
            },
        )
    }
}

impl Default for SymplecticEulerRayon {
    /// Identical to [`SymplecticEulerRayon::new`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Display for SymplecticEulerRayon {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Euler integrator using rayon")
    }
}

impl<State, const D: usize> SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>
    for SymplecticEulerRayon
where
    State: SimulationStateSynchronous<D> + LatticeStateWithEField<D> + LatticeStateWithEFieldNew<D>,
{
    type Error = State::Error;

    #[inline]
    fn integrate_sync_sync(&self, l: &State, delta_t: Real) -> Result<State, Self::Error> {
        let link_matrix = Self::link_matrix_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t,
        );
        let e_field =
            Self::e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t);

        State::new(
            l.lattice().clone(),
            l.beta(),
            EField::new(e_field),
            LinkMatrix::new(link_matrix),
            l.t() + 1,
        )
    }

    #[inline]
    fn integrate_leap_leap(
        &self,
        l: &SimulationStateLeap<State, D>,
        delta_t: Real,
    ) -> Result<SimulationStateLeap<State, D>, Self::Error> {
        let link_matrix = LinkMatrix::new(Self::link_matrix_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t,
        ));

        let e_field = EField::new(Self::e_field_integrate::<State, D>(
            &link_matrix,
            l.e_field(),
            l.lattice(),
            delta_t,
        ));
        SimulationStateLeap::<State, D>::new(
            l.lattice().clone(),
            l.beta(),
            e_field,
            link_matrix,
            l.t() + 1,
        )
    }

    #[inline]
    fn integrate_sync_leap(
        &self,
        l: &State,
        delta_t: Real,
    ) -> Result<SimulationStateLeap<State, D>, Self::Error> {
        let e_field = Self::e_field_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t / 2_f64,
        );

        // we do not advance the time counter
        SimulationStateLeap::<State, D>::new(
            l.lattice().clone(),
            l.beta(),
            EField::new(e_field),
            l.link_matrix().clone(),
            l.t(),
        )
    }

    #[inline]
    fn integrate_leap_sync(
        &self,
        l: &SimulationStateLeap<State, D>,
        delta_t: Real,
    ) -> Result<State, Self::Error> {
        let link_matrix = LinkMatrix::new(Self::link_matrix_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t,
        ));
        // we advance the counter by one
        let e_field = EField::new(Self::e_field_integrate::<State, D>(
            &link_matrix,
            l.e_field(),
            l.lattice(),
            delta_t / 2_f64,
        ));
        State::new(
            l.lattice().clone(),
            l.beta(),
            e_field,
            link_matrix,
            l.t() + 1,
        )
    }

    #[inline]
    fn integrate_symplectic(&self, l: &State, delta_t: Real) -> Result<State, Self::Error> {
        // override for optimization.
        // This remove a clone operation.

        let e_field_half = EField::new(Self::e_field_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t / 2_f64,
        ));
        let link_matrix = LinkMatrix::new(Self::link_matrix_integrate::<State, D>(
            l.link_matrix(),
            &e_field_half,
            l.lattice(),
            delta_t,
        ));
        let e_field = EField::new(Self::e_field_integrate::<State, D>(
            &link_matrix,
            &e_field_half,
            l.lattice(),
            delta_t / 2_f64,
        ));

        State::new(
            l.lattice().clone(),
            l.beta(),
            e_field,
            link_matrix,
            l.t() + 1,
        )
    }
}
