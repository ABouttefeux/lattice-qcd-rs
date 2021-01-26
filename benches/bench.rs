extern crate nalgebra as na;


use lattice_qcd_rs::{
    field::*,
    Real,
    su3::su3_exp_i,
    su3::su3_exp_r,
    lattice::*,
};
use std::{
    f64,
    collections::HashMap,
    vec::Vec,
};

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_simulation_creation_deterministe(
    size: usize,
    rng: &mut rand::rngs::ThreadRng,
    d: &impl rand_distr::Distribution<Real>,
) {
    let _simulation = LatticeSimulationState::new_deterministe(1_f64, size, rng, d).unwrap();
}

fn bench_simulation_creation_threaded<D>(
    size: usize,
    d: &D,
    number_of_thread: usize,
)
    where D: rand_distr::Distribution<Real> + Sync,
{
    let _simulation = LatticeSimulationState::new_random_threaded(1_f64, size, d, number_of_thread).unwrap();
}


fn matrix_exp_old(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>) {
    let _m = (Su3Adjoint::random(rng, d).to_matrix() * na::Complex::<Real>::i()).exp();
}

fn matrix_exp_i(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>) {
    let _m = su3_exp_i(Su3Adjoint::random(rng, d));
}

fn matrix_exp_r(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>) {
    let _m = su3_exp_r(Su3Adjoint::random(rng, d));
}

fn create_su3_adj(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>) {
    let _m = Su3Adjoint::random(rng, d);
}


fn create_vec(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>){
    let l = LatticeCyclique::new(1_f64, 100).unwrap();
    let mut data = Vec::with_capacity(l.get_number_of_points());
    for _ in l.get_points(0) {
        let p1 = Su3Adjoint::random(rng, d);
        let p2 = Su3Adjoint::random(rng, d);
        let p3 = Su3Adjoint::random(rng, d);
        let p4 = Su3Adjoint::random(rng, d);
        data.push(na::Vector4::new(p1, p2, p3, p4));
    }
}

fn create_hash_map(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>){
    let l = LatticeCyclique::new(1_f64, 100).unwrap();
    let mut data = HashMap::with_capacity(l.get_number_of_points());
    for i in l.get_points(0) {
        let p1 = Su3Adjoint::random(rng, d);
        let p2 = Su3Adjoint::random(rng, d);
        let p3 = Su3Adjoint::random(rng, d);
        let p4 = Su3Adjoint::random(rng, d);
        data.insert(i, na::Vector4::new(p1, p2, p3, p4));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    //c.bench_function("sim creation 50", |b| b.iter(|| bench_simulation_creation(50)));
    c.bench_function("sim creation 10 deterministe", |b| {
        b.iter(|| bench_simulation_creation_deterministe(10, &mut rng, &d))
    });
    c.bench_function("sim creation 20 deterministe", |b| {
        b.iter(|| bench_simulation_creation_deterministe(20, &mut rng, &d))
    });
    c.bench_function("sim creation 20 threaded 4", |b| {
        b.iter(|| bench_simulation_creation_threaded(20, &d, 4))
    });
    c.bench_function("sim creation 20 threaded 8", |b| {
        b.iter(|| bench_simulation_creation_threaded(20, &d, 8))
    });
    c.bench_function("sim creation 50 threaded 8", |b| {
        b.iter(|| bench_simulation_creation_threaded(50, &d, 8))
    });
    c.bench_function("matrix exp old", |b| {
        b.iter(|| matrix_exp_old(&mut rng, &d))
    });
    c.bench_function("matrix exp new i ", |b| {
        b.iter(|| matrix_exp_i(&mut rng, &d))
    });
    c.bench_function("matrix exp new r", |b| {
        b.iter(|| matrix_exp_r(&mut rng, &d))
    });
    c.bench_function("creation time random SU3 adj", |b| b.iter(|| create_su3_adj(&mut rng, &d)));
    c.bench_function("creation time Vec", |b| b.iter(|| create_vec(&mut rng, &d)));
    c.bench_function("creation time HashMap", |b| b.iter(|| create_hash_map(&mut rng, &d)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
