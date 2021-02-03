use lattice_qcd_rs::{
    su3,
    simulation::*,
    lattice::*,
    field::*,
    CMatrix3,
};

use std::{
    time::Instant,
    f64,
    vec::Vec
};

use rand::{
    SeedableRng,
    rngs::StdRng
};
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;


fn main() {
    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(0);
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let size = 0.001_f64;
    let number_of_pts = 50;
    let simulation = LatticeHamiltonianSimulationStateSync::new_deterministe(size, 1_f64, number_of_pts, &mut rng, &distribution).unwrap();
    let simulation2 = LatticeHamiltonianSimulationStateSync::new_deterministe_cold_e_hot_link(size, 1_f64, number_of_pts, &mut rng, &distribution).unwrap();
    let sum_gauss = simulation.lattice().get_points().collect::<Vec<LatticePoint>>().into_par_iter().map(|el| {
        simulation.get_gauss(&el).unwrap()
    }).sum::<CMatrix3>();
    let sum_gauss2 = simulation2.lattice().get_points().collect::<Vec<LatticePoint>>().into_par_iter().map(|el| {
        simulation2.get_gauss(&el).unwrap()
    }).sum::<CMatrix3>();
    println!("sum_gauss: {}, sum_gauss2: {}", sum_gauss.norm(), sum_gauss2.norm());
    println!("{:?}", t.elapsed());
}
