

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
    f64,
    // vec::Vec
};

use na::ComplexField;
use na::base::Unit;

use rand::{
    SeedableRng,
    rngs::StdRng
};
// use rayon::iter::IntoParallelIterator;
// use rayon::prelude::ParallelIterator;

use indicatif::{ProgressBar, ProgressStyle};


fn main() {
    //println!("{:?}", m);
    sim_1();
}

fn sim_1() {
    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(0);
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let size = 1000_f64;
    let number_of_pts = 10;
    let beta = 1_f64;
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().tick_chars("|/-\\").template(
        "{prefix:10} [{elapsed_precise}] [{spinner}]"
    ));
    spinner.set_prefix("Generating");
    spinner.tick();
    spinner.enable_steady_tick(200);
    //let mut simulation = LatticeStateDefault::new_deterministe(size, beta, number_of_pts, &mut rng, &distribution).unwrap();
    let mut simulation = LatticeStateDefault::new_cold(size, beta, number_of_pts).unwrap();
    spinner.finish();
    
    println!("initial plaquette avarage {}", simulation.average_trace_plaquette().unwrap());
    
    let delta_t = 0.1_f64;
    let number_of_step = 10;
    let mut hmc = HybridMonteCarlo::new(delta_t, number_of_step, SymplecticEulerRayon::new(), rng);
    let number_of_sims = 2000;
    
    let pb = ProgressBar::new(number_of_sims);
    pb.set_style(ProgressStyle::default_bar().progress_chars("=>-").template(
        "{prefix:10} [{elapsed_precise}] [{bar:40.white/cyan}] {pos:>4}/{len:4} [ETA {eta_precise}] {msg}"
    ));
    pb.set_prefix("Thermalisation");
    pb.tick();
    
    let lattice_link = simulation.lattice().get_link(LatticePoint::from([0,1,2]), Direction::XPos);
    for _ in 0..number_of_sims {
        simulation = simulation.monte_carlo_step(&mut hmc).unwrap();
        let link = simulation.link_matrix().get_matrix(&lattice_link, simulation.lattice()).unwrap();
        println!("P det {}, hermitian error {}", (link.determinant() - Complex::from(1_f64)).modulus(), (link - link.adjoint()).norm());
        simulation.normalize_link_matrices();
        let link = simulation.link_matrix().get_matrix(&lattice_link, simulation.lattice()).unwrap();
        println!("N det {}, hermitian error {}", (link.determinant() - Complex::from(1_f64)).modulus(), (link - link.adjoint()).norm());
        let average = simulation.average_trace_plaquette().unwrap().modulus();
        //pb.set_message(&format!("{}", average));
        //pb.inc(1);
    }
    pb.finish();
    println!("final plaquette avarage {}", simulation.average_trace_plaquette().unwrap());
    println!("{:?}", t.elapsed());
}
