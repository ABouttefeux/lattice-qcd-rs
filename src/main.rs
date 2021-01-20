use lattice_qcd_rs::{
    lattice::*,
    su3::*,
    field::*,
};

use std::{
    time::Instant,
    f64,
};

fn get_n(){
    println!("{:?}",ExponentialSu3::new().n());
}

fn main() {
    
    let t = Instant::now();
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let mut simulation = LatticeSimulation::new(1_f64 , 50, &mut rng, &distribution).unwrap();
    println!("{:?}", t.elapsed());
    get_n();
}
