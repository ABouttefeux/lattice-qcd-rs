//! Simulation module. Contains Monte Carlo algorithms and simulation states.
//!
//! See submodule documentation [`monte_carlo`] and [`state`] for more details.
// TODO more doc

pub mod monte_carlo;
pub mod state;

pub use monte_carlo::*;
pub use state::*;
