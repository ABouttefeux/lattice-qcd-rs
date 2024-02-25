//! Basic symplectic Euler integrator.
//!
//! For an example see the module level documentation [`super`].

use std::{
    error,
    fmt::{self, Display},
};

use nalgebra::SVector;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{integrate_efield, integrate_link, SymplecticIntegrator};
use crate::{
    field::{EField, LinkMatrix, Su3Adjoint},
    lattice::LatticeCyclic,
    simulation::{
        LatticeState, LatticeStateWithEField, LatticeStateWithEFieldNew, SimulationStateLeap,
        SimulationStateSynchronous,
    },
    thread::{run_pool_parallel_vec, ThreadError},
    CMatrix3, Real,
};

/// Error for [`SymplecticEuler`].
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum SymplecticEulerError<Error> {
    /// multithreading error, see [`ThreadError`].
    ThreadingError(ThreadError),
    /// Other Error cause in non threaded section
    StateInitializationError(Error),
}

impl<Error> From<ThreadError> for SymplecticEulerError<Error> {
    #[inline]
    fn from(err: ThreadError) -> Self {
        Self::ThreadingError(err)
    }
}

impl<Error: Display> fmt::Display for SymplecticEulerError<Error> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadingError(error) => write!(f, "thread error: {error}"),
            Self::StateInitializationError(error) => write!(f, "initialization error: {error}"),
        }
    }
}

impl<Error: error::Error + 'static> error::Error for SymplecticEulerError<Error> {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::ThreadingError(error) => Some(error),
            Self::StateInitializationError(error) => Some(error),
        }
    }
}

/// Basic symplectic Euler integrator
///
/// slightly slower than [`super::SymplecticEulerRayon`] (for appropriate choice of `number_of_thread`)
/// but use less memory
///
/// For an example see the module level documentation [`super`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SymplecticEuler {
    /// The number of thread the integrator uses.
    number_of_thread: usize,
}

impl SymplecticEuler {
    getter_copy!(
        /// getter on the number of thread the integrator use.
        #[inline]
        #[must_use]
        pub const,
        number_of_thread,
        usize
    );

    /// Create a integrator using a set number of threads
    #[inline]
    #[must_use]
    pub const fn new(number_of_thread: usize) -> Self {
        Self { number_of_thread }
    }

    fn link_matrix_integrate<State, const D: usize>(
        self,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
        delta_t: Real,
    ) -> Result<Vec<CMatrix3>, ThreadError>
    where
        State: LatticeStateWithEField<D>,
    {
        let result = run_pool_parallel_vec(
            lattice.get_links(),
            &(link_matrix, e_field, lattice),
            &|link, (link_matrix, e_field, lattice)| {
                integrate_link::<State, D>(link, link_matrix, e_field, lattice, delta_t)
            },
            self.number_of_thread,
            lattice.number_of_canonical_links_space(),
            lattice,
            &CMatrix3::zeros(),
        );

        Ok(result?)
    }

    fn e_field_integrate<State, const D: usize>(
        self,
        link_matrix: &LinkMatrix,
        e_field: &EField<D>,
        lattice: &LatticeCyclic<D>,
        delta_t: Real,
    ) -> Result<Vec<SVector<Su3Adjoint, D>>, ThreadError>
    where
        State: LatticeStateWithEField<D>,
    {
        let result = run_pool_parallel_vec(
            lattice.get_points(),
            &(link_matrix, e_field, lattice),
            &|point, (link_matrix, e_field, lattice)| {
                integrate_efield::<State, D>(point, link_matrix, e_field, lattice, delta_t)
            },
            self.number_of_thread,
            lattice.number_of_points(),
            lattice,
            &SVector::<_, D>::from_element(Su3Adjoint::default()),
        );

        Ok(result?)
    }
}

impl Default for SymplecticEuler {
    /// Default value using the number of threads rayon would use,
    /// see [`rayon::current_num_threads()`].
    #[inline]
    fn default() -> Self {
        Self::new(rayon::current_num_threads().min(1))
    }
}

impl Display for SymplecticEuler {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Euler integrator with {} thread",
            self.number_of_thread()
        )
    }
}

impl<State, const D: usize> SymplecticIntegrator<State, SimulationStateLeap<State, D>, D>
    for SymplecticEuler
where
    State: SimulationStateSynchronous<D> + LatticeStateWithEField<D> + LatticeStateWithEFieldNew<D>,
{
    type Error = SymplecticEulerError<State::Error>;

    #[inline]
    fn integrate_sync_sync(&self, l: &State, delta_t: Real) -> Result<State, Self::Error> {
        let link_matrix = self.link_matrix_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t,
        )?;
        let e_field =
            self.e_field_integrate::<State, D>(l.link_matrix(), l.e_field(), l.lattice(), delta_t)?;

        State::new(
            l.lattice().clone(),
            l.beta(),
            EField::new(e_field),
            LinkMatrix::new(link_matrix),
            l.t() + 1,
        )
        .map_err(SymplecticEulerError::StateInitializationError)
    }

    #[inline]
    fn integrate_leap_leap(
        &self,
        l: &SimulationStateLeap<State, D>,
        delta_t: Real,
    ) -> Result<SimulationStateLeap<State, D>, Self::Error> {
        let link_matrix = LinkMatrix::new(self.link_matrix_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t,
        )?);
        let e_field = EField::new(self.e_field_integrate::<State, D>(
            &link_matrix,
            l.e_field(),
            l.lattice(),
            delta_t,
        )?);
        SimulationStateLeap::<State, D>::new(
            l.lattice().clone(),
            l.beta(),
            e_field,
            link_matrix,
            l.t() + 1,
        )
        .map_err(SymplecticEulerError::StateInitializationError)
    }

    #[inline]
    fn integrate_sync_leap(
        &self,
        l: &State,
        delta_t: Real,
    ) -> Result<SimulationStateLeap<State, D>, Self::Error> {
        let e_field = self.e_field_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t / 2_f64,
        )?;
        // we do not advance the step counter
        SimulationStateLeap::<State, D>::new(
            l.lattice().clone(),
            l.beta(),
            EField::new(e_field),
            l.link_matrix().clone(),
            l.t(),
        )
        .map_err(SymplecticEulerError::StateInitializationError)
    }

    #[inline]
    fn integrate_leap_sync(
        &self,
        l: &SimulationStateLeap<State, D>,
        delta_t: Real,
    ) -> Result<State, Self::Error> {
        let link_matrix = LinkMatrix::new(self.link_matrix_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t,
        )?);
        // we advance the counter by one
        let e_field = EField::new(self.e_field_integrate::<State, D>(
            &link_matrix,
            l.e_field(),
            l.lattice(),
            delta_t / 2_f64,
        )?);
        State::new(
            l.lattice().clone(),
            l.beta(),
            e_field,
            link_matrix,
            l.t() + 1,
        )
        .map_err(SymplecticEulerError::StateInitializationError)
    }

    #[inline]
    fn integrate_symplectic(&self, l: &State, delta_t: Real) -> Result<State, Self::Error> {
        // override for optimization.
        // This remove a clone operation.

        let e_field_half = EField::new(self.e_field_integrate::<State, D>(
            l.link_matrix(),
            l.e_field(),
            l.lattice(),
            delta_t / 2_f64,
        )?);
        let link_matrix = LinkMatrix::new(self.link_matrix_integrate::<State, D>(
            l.link_matrix(),
            &e_field_half,
            l.lattice(),
            delta_t,
        )?);
        let e_field = EField::new(self.e_field_integrate::<State, D>(
            &link_matrix,
            &e_field_half,
            l.lattice(),
            delta_t / 2_f64,
        )?);

        State::new(
            l.lattice().clone(),
            l.beta(),
            e_field,
            link_matrix,
            l.t() + 1,
        )
        .map_err(SymplecticEulerError::StateInitializationError)
    }
}
