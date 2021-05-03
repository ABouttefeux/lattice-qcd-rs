use std::error::Error;

use rand::SeedableRng;

use crate::{error::*, integrator::*, simulation::monte_carlo::*, simulation::state::*};

const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

#[test]
fn integrator() -> Result<(), Box<dyn Error>> {
    const DT: f64 = 0.000_1_f64;
    let mut state = LatticeStateDefault::<3>::new_cold(1_f64, 8_f64, 4)?;

    let rng = rand::rngs::StdRng::seed_from_u64(SEED_RNG);
    let mut mh = MetropolisHastingsSweep::new(1, 0.1_f64, rng)
        .ok_or(ImplementationError::OptionWithUnexpectedNone)?;

    for _ in 0..10 {
        state = state.monte_carlo_step(&mut mh)?;
    }

    let mut rng = mh.rng_owned();

    let state_with_e = LatticeStateWithEFieldSyncDefault::new_random_e(
        state.lattice().clone(),
        state.beta(),
        state.link_matrix_owned(),
        &mut rng,
    )?;

    let integrator = SymplecticEulerRayon::new();
    let mut state_new = state_with_e.clone();
    let h = state_new.get_hamiltonian_total();
    state_new = state_new.simulate_symplectic_n(&integrator, DT, 10)?;
    let h2 = state_new.get_hamiltonian_total();
    assert!((h - h2).abs() < 0.000_1_f64);

    let state_new = state_with_e.clone();
    let state_new = state_new.simulate_to_leapfrog(&integrator, DT)?;
    let state_new = state_new.simulate_leap_n(&integrator, DT, 1)?;
    let state_new = state_new.simulate_to_synchrone(&integrator, DT)?;
    let state_new = state_new.simulate_sync_n(&integrator, DT, 1)?;
    let h2 = state_new.get_hamiltonian_total();
    assert!((h - h2).abs() < 0.000_01_f64);

    let state_new = state_with_e.clone();
    let state_new = state_new.simulate_using_leapfrog_n_auto(&integrator, DT, 10)?;
    let h2 = state_new.get_hamiltonian_total();
    assert!((h - h2).abs() < 0.000_01_f64);

    let integrator = SymplecticEuler::default();
    let mut state_new = state_with_e.clone();
    let h = state_new.get_hamiltonian_total();
    state_new = state_new.simulate_symplectic_n(&integrator, DT, 10)?;
    let h2 = state_new.get_hamiltonian_total();
    assert!((h - h2).abs() < 0.000_1_f64);

    let state_new = state_with_e.clone();
    let state_new = state_new.simulate_to_leapfrog(&integrator, DT)?;
    let state_new = state_new.simulate_leap_n(&integrator, DT, 2)?;
    let state_new = state_new.simulate_to_synchrone(&integrator, DT)?;
    let state_new = state_new.simulate_sync_n(&integrator, DT, 2)?;
    let h2 = state_new.get_hamiltonian_total();
    assert!((h - h2).abs() < 0.000_01_f64);

    Ok(())
}
