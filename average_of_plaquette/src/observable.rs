


use lattice_qcd_rs::{
    simulation::*,
    ComplexField,
    lattice::{Direction, DirectionList, LatticePoint},
    dim::{U3},
};
use rayon::prelude::*;

pub fn volume_obs(p: &LatticePoint<U3>, state: &LatticeStateDefault<U3>) -> f64
{
    state.link_matrix().get_pij(&p, &Direction::<U3>::get_all_positive_directions()[0], &Direction::get_all_positive_directions()[1], state.lattice()).map(|el| 1_f64 - el.trace().real() / 3_f64).unwrap()
}

pub fn volume_obs_mean(state: &LatticeStateDefault<U3>) -> f64 {
    let sum = state.lattice().get_points().par_bridge().map(|point| {
        volume_obs(&point, state)
    }).sum::<f64>();
    let number_of_plaquette = state.lattice().get_number_of_points() as f64;
    
    parameter_volume(sum / number_of_plaquette, state.beta())
}

#[allow(clippy::suboptimal_flops)] // readability
pub fn parameter_volume (value: f64, beta: f64) -> f64 {
    let c1: f64 = 8_f64 / 3_f64;
    const C2: f64 = 1.951_315_f64;
    const C3: f64 = 6.861_2_f64;
    const C4: f64 = 2.929_421_32_f64;
    beta.powi(4) * ( value - c1 / beta - C2 / beta.powi(2) - C3 / beta.powi(3) - C4 * beta.ln() / beta.powi(4))
}

type LeapFrogState<D> = SimulationStateLeap<LatticeHamiltonianSimulationStateSyncDefault<LatticeStateDefault<D>, D>, D>;

pub fn e_correletor(state: &LeapFrogState<U3>, state_new: &LeapFrogState<U3>, pt: &LatticePoint<U3>) -> f64 {
    state_new.e_field().get_e_vec(pt, state.lattice()).unwrap()
        .map(|el| el.to_matrix())
        .dot(
            &state_new.e_field().get_e_vec(pt, state.lattice()).unwrap()
                .map(|el| el.to_matrix())
        )
        .trace()
        .real() / (-3_f64)
}
