extern crate nalgebra as na;


use lattice_qcd_rs::{
    field::*,
    Real,
    su3::su3_exp_i,
    su3::su3_exp_r,
};
use std::{
    f64,
};

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_simulation_creation_deterministe(
    size: usize,
    rng: &mut rand::rngs::ThreadRng,
    d: &impl rand_distr::Distribution<Real>,
) {
    let _simulation = LatticeSimulation::new_deterministe(1_f64, size, rng, d).unwrap();
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

fn create_matrix(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>) {
    let _m = Su3Adjoint::random(rng, d);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    //c.bench_function("sim creation 50", |b| b.iter(|| bench_simulation_creation(50)));
    c.bench_function("sim creation 20 deterministe", |b| {
        b.iter(|| bench_simulation_creation_deterministe(20, &mut rng, &d))
    });
    c.bench_function("sim creation 10 deterministe", |b| {
        b.iter(|| bench_simulation_creation_deterministe(10, &mut rng, &d))
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
    c.bench_function("creation time", |b| b.iter(|| create_matrix(&mut rng, &d)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
