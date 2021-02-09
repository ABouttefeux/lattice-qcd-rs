

extern crate lattice_qcd_rs;
extern crate nalgebra as na;
extern crate rand;
extern crate rand_distr;
extern crate indicatif;


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

use dialoguer::{
    Select,
    theme::ColorfulTheme
};

fn main() {
    let items = vec!["Hybrid Monte Carlo", "Metropolis Hastings"];
    let selection = Select::with_theme(&ColorfulTheme::default())
        .items(&items)
        .with_prompt("Choose an algorithm.")
        .default(0)
        .interact_opt().unwrap();

    match selection {
        Some(0) => sim_1(),
        Some(1) => sim_2(),
        _ => println!("no selection")
    }
}

fn generate_state_with_logs(rng: &mut impl rand::Rng) -> LatticeStateDefault {
    let size = 1000_f64;
    let number_of_pts = 5;
    let beta = 1E+2_f64;
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
                simulation.normalize_link_matrices();
            }
            pb.set_message(&closure_message(&simulation, mc));
            pb.inc(sub_block);
        }
    }
    
    pb.finish();
    h.join().unwrap();
    simulation
}

#[allow(dead_code)]
fn sim_1() {
    let t = Instant::now();
    let mut rng_seeder = rand::thread_rng();
    let seed = rng_seeder.next_u64();
    println!("Begining simulation HMC with seed {}", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let delta_t = 0.0075_f64;
    let number_of_step = 100;
    //let mut hmc = HybridMonteCarlo::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let mut hmc = HybridMonteCarloDiagnostic::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let number_of_sims = 100;
    
    let simulation = simulate_loop_with_input(simulation, &mut hmc, number_of_sims, 1,
        &|sim, mc : &HybridMonteCarloDiagnostic<LatticeStateDefault, StdRng, SymplecticEulerRayon>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2},R {}", average, mc.prob_replace_last(), mc.has_replace_last())
        }
    );
    
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}

#[allow(dead_code)]
fn sim_2() {
    let t = Instant::now();
    let mut rng_seeder = rand::thread_rng();
    let seed = rng_seeder.next_u64();
    println!("Begining simulation MH with seed {}", seed);
    let mut rng = StdRng::seed_from_u64(seed);
    //let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let simulation = generate_state_with_logs(&mut rng);
    
    println!("initial plaquette average {}", simulation.average_trace_plaquette().unwrap());
    
    let spread_parameter = 0.001;
    let number_of_rand = 20;
    //let mut mh = MCWrapper::new(MetropolisHastings::new(number_of_rand, spread_parameter).unwrap(), rng);
    let mut mh = MCWrapper::new(MetropolisHastingsDiagnostic::new(number_of_rand, spread_parameter).unwrap(), rng);
    let number_of_sims = 40000;
    
    //let simulation = simulate_with_log_subblock(simulation, &mut mh, number_of_sims);
    
    let simulation = simulate_loop_with_input(simulation, &mut mh, number_of_sims, 1000,
        &|sim, mc : &MCWrapper<MetropolisHastingsDiagnostic<LatticeStateDefault>, LatticeStateDefault, StdRng>| {
            let average = sim.average_trace_plaquette().unwrap().real();
            format!("A {:.6},P {:.2},R {}", average, mc.mcd().prob_replace_last(), mc.mcd().has_replace_last())
        }
    );
    //let simulation = simulate_loop_with_input_diag_mh(simulation, &mut mh, number_of_sims, 1000);

        
    println!("final plaquette average {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}
