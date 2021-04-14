
//! Some code to test the library

extern crate lattice_qcd_rs;
extern crate nalgebra as na;
extern crate rand;
extern crate rand_distr;
extern crate indicatif;
extern crate bincode;
extern crate rand_xoshiro;
extern crate rayon;

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

use na::{
    ComplexField,
    SVector,
};

use rand::{
    SeedableRng,
};
// use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use rand::RngCore;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Read};
#[cfg(test)]
use std::fs::File;
use std::sync::*;
use std::thread;
#[cfg(test)]
use std::io::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use dialoguer::{
    Select,
    theme::ColorfulTheme
};

const DIM_USE: usize = 3;
const NB_OF_PTS: usize = 10;
const BETA: f64 = 24_f64;

fn main() {
    let items = vec!["Hybrid Monte Carlo", "Metropolis Hastings absolute", "Metropolis Hastings Delta", "MHD + HMC"];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .items(&items)
        .with_prompt("Choose an algorithm.")
        .default(0)
        .interact_opt().unwrap();

    match selection {
        Some(0) => sim_hmc(),
        Some(1) => sim_mh(),
        Some(2) => sim_dmh(),
        Some(3) => sim_dmh_hmc(),
        None => println!("no selection"),
        _ => unreachable!(),
    }
}

#[cfg(test)]
#[test]
#[ignore]
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
    let decoded: LatticeStateDefault<4> = bincode::deserialize(&encoded_2).unwrap();
    println!("decoded");
    assert_eq!(decoded, state);
    Ok(())
}

#[cfg(test)]
#[test]
fn test_leap_frog() {
    let mut rng = Xoshiro256PlusPlus::from_entropy();
    let size = 1000_f64;
    let number_of_pts = 4;
    let beta = 1_f64;
    let state = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, &mut rng).unwrap();
    println!("h_l {}", state.get_hamiltonian_links());
    let state_hmc = LatticeStateWithEFieldSyncDefault::<LatticeStateDefault<4>, 4>::new_random_e_state(state, &mut rng);
    let h1 = state_hmc.get_hamiltonian_total();
    println!("h_t {}", h1);
    let state_hmc_2 = state_hmc.simulate_using_leapfrog_n_auto(&SymplecticEulerRayon::new(), 0.01, 1).unwrap();
    let h2 = state_hmc_2.get_hamiltonian_total();
    println!("h_t {}", h2);
    println!("{}", (h1-h2).exp() );

    assert!((h1-h2).abs() < 0.000_01);
}

fn generate_state_with_logs<const D: usize>(rng: &mut impl rand::Rng) -> LatticeStateDefault<D>
    where
    SVector<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    let size = 1000_f64;
    let number_of_pts = NB_OF_PTS;
    let beta = BETA;
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

#[allow(dead_code)]
fn simulate_with_log_subblock<MC>(mut simulation: LatticeStateDefault<4>, mc: &mut MC, number_of_sims: u64) -> LatticeStateDefault<4>
    where MC: MonteCarlo<LatticeStateDefault<4>, 4>,
    MC::Error: core::fmt::Debug,
{
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

#[allow(clippy::mutex_atomic)]
fn simulate_loop_with_input<MC, const D: usize>(
    mut simulation: LatticeStateDefault<D>,
    mc: &mut MC,
    number_of_sims: u64,
    sub_block: u64,
    closure_message : &dyn Fn(&LatticeStateDefault<D>, &MC) -> String
) -> LatticeStateDefault<D>
    where MC: MonteCarlo<LatticeStateDefault<D>, D>,
    SVector<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
    MC::Error: core::fmt::Debug,
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
        let _ = stdin.read(&mut [0u8; 4]).unwrap();
        *stop_sim_c.lock().unwrap() = true;
    });
    
    while ! *stop_sim.lock().unwrap() {
        pb.reset_eta();
        pb.set_position(0);
        for _ in 0..number_of_sims/sub_block {
            for _ in 0..sub_block {
                simulation = simulation.monte_carlo_step(mc).unwrap();
                pb.inc(1);
            }
            simulation.normalize_link_matrices();
            pb.set_message(&closure_message(&simulation, mc));
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
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation: LatticeStateDefault<DIM_USE> = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let delta_t = 0.1_f64;
    let number_of_step = 100;
    //let mut hmc = HybridMonteCarlo::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let mut hmc = HybridMonteCarloDiagnostic::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let number_of_sims = 10;
    
    let simulation = simulate_loop_with_input(simulation, &mut hmc, number_of_sims, 1,
        &|sim, mc : &HybridMonteCarloDiagnostic<LatticeStateDefault<DIM_USE>, Xoshiro256PlusPlus, SymplecticEulerRayon, DIM_USE>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2},R {}", average / 3_f64, mc.prob_replace_last(), mc.has_replace_last())
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
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation: LatticeStateDefault<DIM_USE> = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let spread_parameter = 0.00001;
    let number_of_rand = 1;
    //let mut mh = MCWrapper::new(MetropolisHastings::new(number_of_rand, spread_parameter).unwrap(), rng);
    let mut mh = McWrapper::new(MetropolisHastingsDiagnostic::new(number_of_rand, spread_parameter).unwrap(), rng);
    let number_of_sims = 1000;
    
    //let simulation = simulate_with_log_subblock(simulation, &mut mh, number_of_sims);
    
    let simulation = simulate_loop_with_input(simulation, &mut mh, number_of_sims, 100,
        &|sim, mc : &McWrapper<MetropolisHastingsDiagnostic, LatticeStateDefault<DIM_USE>, Xoshiro256PlusPlus, DIM_USE>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2},R {}", average / 3_f64, mc.mcd().prob_replace_last(), mc.mcd().has_replace_last())
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
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let simulation: LatticeStateDefault<DIM_USE> = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let spread_parameter = 0.5;
    //let mut mh = MCWrapper::new(MetropolisHastings::new(number_of_rand, spread_parameter).unwrap(), rng);
    let mut mh = MetropolisHastingsDeltaDiagnostic::new(spread_parameter, rng).unwrap();
    let number_of_sims = 1000;
    let sub_block = 100;
    
    //let simulation = simulate_with_log_subblock(simulation, &mut mh, number_of_sims);
    
    let simulation = simulate_loop_with_input(simulation, &mut mh, number_of_sims, sub_block,
        &|sim: &LatticeStateDefault<DIM_USE>, mc : &MetropolisHastingsDeltaDiagnostic<Xoshiro256PlusPlus>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2}", average / 3_f64, mc.prob_replace_last())
        }
    );
    //let simulation = simulate_loop_with_input_diag_mh(simulation, &mut mh, number_of_sims, 1000);
    
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}

fn sim_dmh_hmc() {
    let t = Instant::now();
    let mut rng_seeder = rand::thread_rng();
    let seed = rng_seeder.next_u64();
    println!("Begining simulation MH Delta -> HMC with seed {:#08x}", seed);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let simulation: LatticeStateDefault<DIM_USE> = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let spread_parameter = 0.1;
    let mut mh = MetropolisHastingsDeltaDiagnostic::new(spread_parameter, rng).unwrap();
    let number_of_sims = 10_000;
    let sub_block = 1_000;
    
    let initial_data = simulation.lattice().get_points().par_bridge().map(|point| {
        simulation.link_matrix().get_pij(&point, &Direction::<3>::get_all_positive_directions()[0], &Direction::get_all_positive_directions()[1], simulation.lattice()).map(|el| 1_f64 - el.trace().real() / 3_f64).unwrap()
    }).collect::<Vec<f64>>();
    
    let simulation = simulate_loop_with_input(simulation, &mut mh, number_of_sims, sub_block,
        &|sim: &LatticeStateDefault<DIM_USE>, _mc : &MetropolisHastingsDeltaDiagnostic<Xoshiro256PlusPlus>| {
            //let average = sim.average_trace_plaquette().unwrap().real();
            let vec = sim.lattice().get_points().par_bridge().map(|point| {
                sim.link_matrix().get_pij(&point, &Direction::<3>::get_all_positive_directions()[0], &Direction::get_all_positive_directions()[1], sim.lattice()).map(|el| 1_f64 - el.trace().real() / 3_f64).unwrap()
            }).collect::<Vec<f64>>();
            
            let sum = vec.iter().sum::<f64>();
            let number_of_plaquette = sim.lattice().get_number_of_points() as f64;
            let c1 = 8_f64 / 3_f64;
            let c2 = 1.951315_f64;
            let c3 = 6.8612_f64;
            let c4 = 2.92942132_f64;
            let a = sim.beta().powi(4) * ((sum / number_of_plaquette) - c1 / sim.beta() - c2 / sim.beta().powi(2) - c3 / sim.beta().powi(3) - c4 * sim.beta().ln() / sim.beta().powi(4));
            format!("A {:.6},  {}", a, lattice_qcd_rs::statistics::covariance(&initial_data, &vec).unwrap().abs())
        }
    );
    
    let rng = mh.rng_owned();
    let delta_t = 0.01_f64;
    let number_of_step = 200;
    //let mut hmc = HybridMonteCarlo::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let mut hmc = HybridMonteCarloDiagnostic::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let number_of_sims = 10;
    
    let simulation = simulate_loop_with_input(simulation, &mut hmc, number_of_sims, 1,
        &|sim: &LatticeStateDefault<DIM_USE>, mc : &HybridMonteCarloDiagnostic<LatticeStateDefault<DIM_USE>, Xoshiro256PlusPlus, SymplecticEulerRayon, DIM_USE>| {
            //let average = sim.average_trace_plaquette().unwrap().real();
            let sum = sim.lattice().get_points().par_bridge().map(|point| {
                sim.link_matrix().get_pij(&point, &Direction::<3>::get_all_positive_directions()[0], &Direction::get_all_positive_directions()[1], sim.lattice()).map(|el| 1_f64 - el.trace().real() / 3_f64)
            }).sum::<Option<f64>>().unwrap();
            let number_of_plaquette = sim.lattice().get_number_of_points() as f64;
            let c1 = 8_f64 / 3_f64;
            let c2 = 1.951315_f64;
            let c3 = 6.8612_f64;
            let c4 = 2.92942132_f64;
            let a = sim.beta().powi(4) * ((sum / number_of_plaquette) - c1 / sim.beta() - c2 / sim.beta().powi(2) - c3 / sim.beta().powi(3) - c4 * sim.beta().ln() / sim.beta().powi(4));
            format!("A {:.6},P {:.2},R {}", a, mc.prob_replace_last(), mc.has_replace_last())
        }
    );
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}
