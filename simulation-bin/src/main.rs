

extern crate lattice_qcd_rs;
extern crate nalgebra as na;
extern crate rand;
extern crate rand_distr;
extern crate indicatif;
extern crate bincode;

#[allow(unused_imports)]
use lattice_qcd_rs::{
    simulation::*,
    lattice::*,
    field::*,
    CMatrix3,
    integrator::*,
    Real,
    Complex,
    su3::*,
};
use std::{
    time::Instant,
    //f64,
    // vec::Vec
};

use na::ComplexField;

use rand::{
    SeedableRng,
    rngs::StdRng
};
// use rayon::iter::IntoParallelIterator;
// use rayon::prelude::ParallelIterator;
use rand::RngCore;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Read};
use std::sync::*;
use std::thread;
use std::fs::File;
use std::io::prelude::*;

use dialoguer::{
    Select,
    theme::ColorfulTheme
};

fn main() {
    let items = vec!["Hybrid Monte Carlo", "Metropolis Hastings absolute", "Metropolis Hastings Delta"];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .items(&items)
        .with_prompt("Choose an algorithm.")
        .default(0)
        .interact_opt().unwrap();

    match selection {
        Some(0) => sim_hmc(),
        Some(1) => sim_mh(),
        Some(2) => sim_dmh(),
        None => println!("no selection"),
        _ => unreachable!(),
    }
}

fn test_write_read() -> std::io::Result<()> {
    let mut rng = rand::thread_rng();
    let state = generate_state_with_logs(&mut rng);
    let encoded: Vec<u8> = bincode::serialize(&state).unwrap();
    let mut file = File::create("test.txt")?;
    file.write_all(&encoded)?;
    drop(file);
    let mut f = File::open("test.txt")?;
    let mut encoded_2: Vec<u8> = vec![];
    f.read_to_end(&mut encoded_2)?;
    println!("{}", encoded_2.len());
    println!("read");
    let decoded: LatticeStateDefault = bincode::deserialize(&encoded_2).unwrap();
    println!("decoded");
    assert_eq!(decoded, state);
    Ok(())
}

fn test_leap_frog() {
    let mut rng = rand::thread_rng();
    let state = generate_state_with_logs(&mut rng);
    println!("h_l {}", state.get_hamiltonian_links());
    let state_hmc = LatticeHamiltonianSimulationStateSyncDefault::<LatticeStateDefault>::new_random_e_state(state, &mut rng);
    let h1 = state_hmc.get_hamiltonian_total();
    println!("h_t {}", h1);
    let state_hmc_2 = state_hmc.simulate_using_leapfrog_n_auto(0.01, 1, &SymplecticEulerRayon::new()).unwrap();
    let h2 = state_hmc_2.get_hamiltonian_total();
    println!("h_t {}", h2);
    println!("{}", (h1-h2).exp());
}

fn generate_state_with_logs(rng: &mut impl rand::Rng) -> LatticeStateDefault {
    let size = 1000_f64;
    let number_of_pts = 12;
    let beta = 2_f64;
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().tick_chars("|/-\\").template(
        "{prefix:10} [{elapsed_precise}] [{spinner}]"
    ));
    spinner.set_prefix("Generating");
    spinner.tick();
    spinner.enable_steady_tick(200);
    let simulation = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, rng).unwrap();
    //let simulation = LatticeStateDefault::new_cold(size, beta, number_of_pts).unwrap();
    spinner.finish();
    simulation
}

fn simulate_with_log_subblock(mut simulation: LatticeStateDefault, mc: &mut impl MonteCarlo<LatticeStateDefault> , number_of_sims: u64) -> LatticeStateDefault {
    let pb = ProgressBar::new(number_of_sims);
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        "{prefix:10} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>4}/{len:4} [ETA {eta_precise}] {msg}"
    ));
    pb.set_prefix("Thermalisation");
    pb.tick();
    pb.enable_steady_tick(499);
    let sub_block = 1000;
    for _ in 0..number_of_sims/sub_block {
        for _ in 0..sub_block {
            simulation = simulation.monte_carlo_step(mc).unwrap();
            simulation.normalize_link_matrices();
        }
        let average = simulation.average_trace_plaquette().unwrap().real();
        pb.set_message(&format!("{}", average));
        pb.inc(sub_block);
    }
    pb.finish();
    
    simulation
}


fn simulate_loop_with_input<MC>(
    mut simulation: LatticeStateDefault,
    mc: &mut MC,
    number_of_sims: u64,
    sub_block: u64,
    closure_message : &dyn Fn(&LatticeStateDefault, &MC) -> String
) -> LatticeStateDefault
    where MC: MonteCarlo<LatticeStateDefault>
{
    
    println!();
    println!("   |--------------------------------------|");
    println!("   | Press enter to finish the simulation |");
    println!("   |--------------------------------------|");
    println!();
    
    let pb = ProgressBar::new(number_of_sims);
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        "{prefix:10} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>4}/{len:4} [ETA {eta_precise}] {msg}"
    ));
    pb.set_prefix("Thermalisation");
    pb.tick();
    pb.enable_steady_tick(499);
    
    let stop_sim = Arc::new(Mutex::new(false));
    let stop_sim_c = Arc::clone(&stop_sim);
    let h = thread::spawn(move || {
        let mut stdin = io::stdin();
        let _ = stdin.read(&mut [0u8]).unwrap();
        *stop_sim_c.lock().unwrap() = true;
    });
    
    while ! *stop_sim.lock().unwrap() {
        pb.reset_eta();
        pb.set_position(0);
        for _ in 0..number_of_sims/sub_block {
            for _ in 0..sub_block {
                simulation = simulation.monte_carlo_step(mc).unwrap();
            }
            simulation.normalize_link_matrices();
            pb.set_message(&closure_message(&simulation, mc));
            pb.inc(sub_block);
        }
    }
    
    pb.finish();
    h.join().unwrap();
    simulation
}

fn sim_hmc() {
    let t = Instant::now();
    let mut rng_seeder = rand::thread_rng();
    let seed = rng_seeder.next_u64();
    println!("Begining simulation HMC with seed {:#08x}", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let delta_t = 0.075_f64;
    let number_of_step = 100;
    //let mut hmc = HybridMonteCarlo::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let mut hmc = HybridMonteCarloDiagnostic::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let number_of_sims = 10;
    
    let simulation = simulate_loop_with_input(simulation, &mut hmc, number_of_sims, 1,
        &|sim, mc : &HybridMonteCarloDiagnostic<LatticeStateDefault, StdRng, SymplecticEulerRayon>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2},R {}", average, mc.prob_replace_last(), mc.has_replace_last())
        }
    );
    
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}

fn sim_mh() {
    let t = Instant::now();
    let mut rng_seeder = rand::thread_rng();
    let seed = rng_seeder.next_u64();
    println!("Begining simulation MH with seed {:#08x}", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let spread_parameter = 0.00001;
    let number_of_rand = 20;
    //let mut mh = MCWrapper::new(MetropolisHastings::new(number_of_rand, spread_parameter).unwrap(), rng);
    let mut mh = MCWrapper::new(MetropolisHastingsDiagnostic::new(number_of_rand, spread_parameter).unwrap(), rng);
    let number_of_sims = 1000;
    
    //let simulation = simulate_with_log_subblock(simulation, &mut mh, number_of_sims);
    
    let simulation = simulate_loop_with_input(simulation, &mut mh, number_of_sims, 100,
        &|sim, mc : &MCWrapper<MetropolisHastingsDiagnostic<LatticeStateDefault>, LatticeStateDefault, StdRng>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2},R {}", average, mc.mcd().prob_replace_last(), mc.mcd().has_replace_last())
        }
    );
    //let simulation = simulate_loop_with_input_diag_mh(simulation, &mut mh, number_of_sims, 1000);
        
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}

fn sim_dmh() {
    let t = Instant::now();
    let mut rng_seeder = rand::thread_rng();
    let seed = rng_seeder.next_u64();
    println!("Begining simulation MH Delta with seed {:#08x}", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let spread_parameter = 0.00001;
    let number_of_rand = 20;
    //let mut mh = MCWrapper::new(MetropolisHastings::new(number_of_rand, spread_parameter).unwrap(), rng);
    let mut mh = MCWrapper::new(MetropolisHastingsDeltaDiagnostic::new(number_of_rand, spread_parameter).unwrap(), rng);
    let number_of_sims = 1000;
    let sub_block = 100;
    
    //let simulation = simulate_with_log_subblock(simulation, &mut mh, number_of_sims);
    
    let simulation = simulate_loop_with_input(simulation, &mut mh, number_of_sims, sub_block,
        &|sim, mc : &MCWrapper<MetropolisHastingsDeltaDiagnostic, LatticeStateDefault, StdRng>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2}", average, mc.mcd().prob_replace_last())
        }
    );
    //let simulation = simulate_loop_with_input_diag_mh(simulation, &mut mh, number_of_sims, 1000);
        
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}
