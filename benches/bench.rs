extern crate nalgebra as na;


use lattice_qcd_rs::{
    field::*,
    Real,
    su3::su3_exp_i,
    su3::su3_exp_r,
    integrator::*,
    simulation::*,
};
use std::{
    f64,
    collections::HashMap,
    vec::Vec,
};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn bench_simulation_creation_deterministe(
    size: usize,
    rng: &mut rand::rngs::ThreadRng,
    d: &impl rand_distr::Distribution<Real>,
) {
    let _simulation = LatticeSimulationStateSync::new_deterministe(1_f64, size, rng, d).unwrap();
}

fn bench_simulation_creation_threaded<D>(
    size: usize,
    d: &D,
    number_of_thread: usize,
)
    where D: rand_distr::Distribution<Real> + Sync,
{
    let _simulation = LatticeSimulationStateSync::new_random_threaded(1_f64, size, d, number_of_thread).unwrap();
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


fn create_vec(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>, size: usize){
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(na::Vector4::new(
            Su3Adjoint::random(rng, d),
            Su3Adjoint::random(rng, d),
            Su3Adjoint::random(rng, d),
            Su3Adjoint::random(rng, d),
        ));
    }
}

fn create_hash_map(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<Real>, size: usize){
    let mut data = HashMap::with_capacity(size);
    for i in 0..size {
        data.insert(i, na::Vector4::new(
            Su3Adjoint::random(rng, d),
            Su3Adjoint::random(rng, d),
            Su3Adjoint::random(rng, d),
            Su3Adjoint::random(rng, d),
        ));
    }
}

fn simulate_euler(simulation: &mut LatticeSimulationStateSync, number_of_thread: usize) {
    *simulation = simulation.simulate::<SymplecticEuler>(0.00001, SymplecticEuler::new(number_of_thread)).unwrap();
}

fn simulate_euler_rayon(simulation: &mut LatticeSimulationStateSync) {
    *simulation = simulation.simulate::<SymplecticEulerRayon>(0.00001, SymplecticEulerRayon::new()).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    
    let mut rng = rand::thread_rng();
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    
    let mut groupe_creation_deterministe = c.benchmark_group("Sim creation deterministe");
    groupe_creation_deterministe.sample_size(10);
    let array_size: [usize; 8] = [2, 4, 8, 10, 20, 30, 50, 75];
    for el in array_size.iter(){
        groupe_creation_deterministe.throughput(Throughput::Elements((el * el * el) as u64));
        groupe_creation_deterministe.bench_with_input(BenchmarkId::new("size", el * el* el), el,
            |b,i| b.iter(|| bench_simulation_creation_deterministe(*i, &mut rng, &d))
        );
    }
    groupe_creation_deterministe.finish();
    
    let mut groupe_creation_threaded = c.benchmark_group("Sim creation (20) threaded");
    let gp_size = 20;
    groupe_creation_threaded.sample_size(50);
    groupe_creation_threaded.throughput(Throughput::Elements((gp_size * gp_size * gp_size) as u64));
    let thread_count: [usize; 5] = [1, 2, 4, 6, 8];
    for n in thread_count.iter(){
        groupe_creation_threaded.bench_with_input(BenchmarkId::new("thread", n), n,
            |b,i| b.iter(|| bench_simulation_creation_threaded(gp_size, &d, *i))
        );
    }
    groupe_creation_threaded.finish();
    
    let mut groupe_matrix = c.benchmark_group("matrix exp");
    groupe_matrix.sample_size(2_000);
    groupe_matrix.bench_function("matrix exp old", |b| {
        b.iter(|| matrix_exp_old(&mut rng, &d))
    });
    groupe_matrix.bench_function("matrix exp new i ", |b| {
        b.iter(|| matrix_exp_i(&mut rng, &d))
    });
    groupe_matrix.bench_function("matrix exp new r", |b| {
        b.iter(|| matrix_exp_r(&mut rng, &d))
    });
    groupe_matrix.finish();
    
    let mut groupe_sim = c.benchmark_group("simulations");
    groupe_sim.sample_size(10);
    let thread_count: [usize; 5] = [1, 2, 4, 6, 8];
    for n in thread_count.iter(){
        let mut sim = LatticeSimulationStateSync::new_random_threaded(1_f64, 20, &d, 4).unwrap();
        groupe_sim.bench_with_input(BenchmarkId::new("thread", n), n,
            |b,i| b.iter(|| simulate_euler(&mut sim, *i))
        );
    }
    
    let mut sim = LatticeSimulationStateSync::new_random_threaded(1_f64, 20, &d, 4).unwrap();
    groupe_sim.bench_function("simulate(20) rayon", |b| {
        b.iter(|| simulate_euler_rayon(&mut sim))
    });
    groupe_sim.finish();
}

fn benchmark_base(c: &mut Criterion) {
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let mut rng = rand::thread_rng();
    
    let mut groupe_collection = c.benchmark_group("std collections");
    groupe_collection.sample_size(10);
    let size: [usize; 5] = [100, 1_000, 10_000, 100_000, 1_000_000];
    for s in size.iter() {
        groupe_collection.throughput(Throughput::Elements(*s as u64));
        groupe_collection.bench_with_input(BenchmarkId::new("vec size", s), s,
            |b,i| b.iter(|| create_vec(&mut rng, &d, *i))
        );
        groupe_collection.bench_with_input(BenchmarkId::new("hash_map size", s), s,
            |b,i| b.iter(|| create_hash_map(&mut rng, &d, *i))
        );
    }
    groupe_collection.finish();
    
    let mut groupe_random = c.benchmark_group("random gen");
    groupe_random.sample_size(2_000);
    groupe_random.bench_function("creation time random SU3 adj", |b| b.iter(|| create_su3_adj(&mut rng, &d)));
    groupe_random.finish();
}


criterion_group!(benches, criterion_benchmark, benchmark_base);
criterion_main!(benches);
