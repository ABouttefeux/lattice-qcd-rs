


use lattice_qcd_rs::{
    simulation::*,
    ComplexField,
    lattice::{Direction, DirectionList, LatticePoint},
    dim::{U3},
};
use rayon::prelude::*;

pub fn volume_obs(p: &LatticePoint<U3>, state: &LatticeStateDefault<U3>) -> f64
{
    let number_of_directions = (Direction::<U3>::dim() * (Direction::<U3>::dim() - 1)) * 2; // ( *4 / 2)
    let directions_all = Direction::<U3>::get_all_directions();
    // We consider all plaquette in positive and negative directions
    // but we avoid counting two times the plaquette P_IJ P_JI
    // as this is manage by taking the real part
    directions_all.iter().map(|dir_1| {
        directions_all.iter()
            .filter(|dir_2| dir_1.to_index() > dir_2.to_index())
            .map(|dir_2| {
                state.link_matrix().get_pij(&p, &dir_1, &dir_2, state.lattice()).map(|el| 1_f64 - el.trace().real() / 3_f64).unwrap()
            }).sum::<f64>()
    }).sum::<f64>() / number_of_directions as f64
    
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


pub fn e_correletor(state: &LatticeHamiltonianSimulationStateSyncDefault<LatticeStateDefault<U3>, U3>, state_new: &LatticeHamiltonianSimulationStateSyncDefault<LatticeStateDefault<U3>, U3>, pt: &LatticePoint<U3>) -> Option<f64> {
    Some(
        state_new.e_field().get_e_vec(pt, state_new.lattice())?.iter()
            .zip(state.e_field().get_e_vec(pt, state.lattice())?.iter())
            .map(|(el1, el2)| {
                el1.iter().zip(el2.iter())
                    .map(|(d1, d2)| (d1 * d2))
                    .sum::<f64>()
            })
            .sum::<f64>() / (2_f64 * 3_f64)
    )
}
