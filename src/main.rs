use lattice_qcd_rs::{
    lattice::*,
    su3::*,
    field::*,
};

use std::{
    time::Instant,
    f64,
};

use rand::{
    SeedableRng,
    rngs::StdRng
};


fn get_n(){
    println!("N is {:}",ExponentialSu3::new().n());
}

fn main() {
    get_n();
    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(0);
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let mut simulation = LatticeSimulation::new_deterministe(1_f64, 50, &mut rng, &distribution).unwrap();
    println!("{:?}", t.elapsed());
    
}
