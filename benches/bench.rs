extern crate nalgebra as na;

use std::{collections::HashMap, f64, vec::Vec};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use lattice_qcd_rs::{
    field::*, integrator::*, simulation::*, su3::su3_exp_i, su3::su3_exp_r, Complex, Real,
};
use rayon::prelude::*;

fn bench_simulation_creation_determinist(
    size: usize,
    rng: &mut rand::rngs::ThreadRng,
    d: &impl rand_distr::Distribution<Real>,
) {
    let _simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(
        1_f64, 1_f64, size, rng, d,
    )
    .unwrap();
}

fn bench_simulation_creation_threaded<D>(size: usize, d: &D, number_of_thread: usize)
where
    D: rand_distr::Distribution<Real> + Sync,
{
    let _simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_random_threaded(
        1_f64,
        1_f64,
        size,
        d,
        number_of_thread,
    )
    .unwrap();
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

fn create_vec(
    rng: &mut rand::rngs::ThreadRng,
    d: &impl rand_distr::Distribution<Real>,
    size: usize,
) {
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

fn create_hash_map(
    rng: &mut rand::rngs::ThreadRng,
    d: &impl rand_distr::Distribution<Real>,
    size: usize,
) {
    let mut data = HashMap::with_capacity(size);
    for i in 0..size {
        data.insert(
            i,
            na::Vector4::new(
                Su3Adjoint::random(rng, d),
                Su3Adjoint::random(rng, d),
                Su3Adjoint::random(rng, d),
                Su3Adjoint::random(rng, d),
            ),
        );
    }
}

fn simulate_euler(
    simulation: &mut LatticeStateEFSyncDefault<LatticeStateDefault<4>, 4>,
    number_of_thread: usize,
) {
    *simulation = simulation
        .simulate_sync(&SymplecticEuler::new(number_of_thread), 0.00001)
        .unwrap();
}

fn simulate_euler_rayon(simulation: &mut LatticeStateEFSyncDefault<LatticeStateDefault<4>, 4>) {
    *simulation = simulation
        .simulate_sync(&SymplecticEulerRayon::new(), 0.00001)
        .unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);

    let mut groupe_creation_determinist = c.benchmark_group("Sim creation determinist");
    groupe_creation_determinist.sample_size(10);
    let array_size: [usize; 6] = [2, 4, 8, 10, 20, 30];
    for el in array_size.iter() {
        groupe_creation_determinist.throughput(Throughput::Elements((el.pow(4)) as u64));
        groupe_creation_determinist.bench_with_input(
            BenchmarkId::new("size", el.pow(4)),
            el,
            |b, i| b.iter(|| bench_simulation_creation_determinist(*i, &mut rng, &d)),
        );
    }
    groupe_creation_determinist.finish();

    let mut groupe_creation_threaded = c.benchmark_group("Sim creation (20) threaded");
    let gp_size: usize = 20;
    groupe_creation_threaded.sample_size(50);
    groupe_creation_threaded.throughput(Throughput::Elements(gp_size.pow(4) as u64));
    let thread_count: [usize; 5] = [1, 2, 4, 6, 8];
    for n in thread_count.iter() {
        groupe_creation_threaded.bench_with_input(BenchmarkId::new("thread", n), n, |b, i| {
            b.iter(|| bench_simulation_creation_threaded(gp_size, &d, *i))
        });
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
    for n in thread_count.iter() {
        let mut sim = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_random_threaded(
            1_f64, 1_f64, 5, &d, 4,
        )
        .unwrap();
        groupe_sim.bench_with_input(BenchmarkId::new("thread", n), n, |b, i| {
            b.iter(|| simulate_euler(&mut sim, *i))
        });
    }

    let mut sim = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_random_threaded(
        1_f64, 1_f64, 5, &d, 4,
    )
    .unwrap();
    groupe_sim.bench_function("simulate(20) rayon", |b| {
        b.iter(|| simulate_euler_rayon(&mut sim))
    });
    groupe_sim.finish();

    let mut groupe_mc = c.benchmark_group("Monte Carlo");
    groupe_mc.sample_size(10);

    let mut mc = MetropolisHastingsSweep::new(1, 0.1_f64, rand::thread_rng()).unwrap();

    groupe_mc.bench_function("simulate 20 D3 Metropolis Hastings Sweep", |b| {
        b.iter_batched(
            || LatticeStateDefault::<3>::new_determinist(1000_f64, 2_f64, 20, &mut rng).unwrap(),
            |state_in| state_in.monte_carlo_step(&mut mc),
            BatchSize::LargeInput,
        )
    });

    let mut mch =
        HybridMonteCarloDiagnostic::new(0.01, 100, SymplecticEulerRayon::new(), rand::thread_rng());

    groupe_mc.bench_function("simulate 20 D3 hybrid monteCarlo 100", |b| {
        b.iter_batched(
            || LatticeStateDefault::<3>::new_determinist(1000_f64, 2_f64, 20, &mut rng).unwrap(),
            |state_in| state_in.monte_carlo_step(&mut mch),
            BatchSize::LargeInput,
        )
    });
    groupe_mc.finish();

    let mut groupe_gauss_proj = c.benchmark_group("Gauss Projection");
    groupe_gauss_proj.sample_size(10);

    let array_size: [usize; 4] = [2, 4, 8, 10];
    for n in array_size.iter() {
        groupe_gauss_proj.bench_with_input(BenchmarkId::new("size", n), n, |b, i| {
            b.iter(|| {
                let state =
                    LatticeStateDefault::<4>::new_determinist(1_f64, 1_f64, *i, &mut rng).unwrap();
                let d = rand_distr::Normal::new(0.0, 0.5_f64).unwrap();
                let e_field = EField::new_determinist(state.lattice(), &mut rng, &d);
                let _ = e_field.project_to_gauss(state.link_matrix(), state.lattice());
            })
        });
    }
    groupe_gauss_proj.finish();

    let mut groupe_clone = c.benchmark_group("Clone");
    groupe_clone.sample_size(100);
    let link_m = LinkMatrix::new(vec![na::Matrix3::<Complex>::identity(); 10_000]);
    groupe_clone.bench_function("rayon clone", |b| {
        b.iter(|| LinkMatrix::new(link_m.data().par_iter().copied().collect()))
    });

    groupe_clone.bench_function("clone", |b| b.iter(|| link_m.clone()));

    groupe_clone.finish()
}

fn benchmark_base(c: &mut Criterion) {
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let mut rng = rand::thread_rng();

    let mut groupe_collection = c.benchmark_group("std collections");
    groupe_collection.sample_size(10);
    let size: [usize; 5] = [100, 1_000, 10_000, 100_000, 1_000_000];
    for s in size.iter() {
        groupe_collection.throughput(Throughput::Elements(*s as u64));
        groupe_collection.bench_with_input(BenchmarkId::new("vec size", s), s, |b, i| {
            b.iter(|| create_vec(&mut rng, &d, *i))
        });
        groupe_collection.bench_with_input(BenchmarkId::new("hash_map size", s), s, |b, i| {
            b.iter(|| create_hash_map(&mut rng, &d, *i))
        });
    }
    groupe_collection.finish();

    let mut groupe_random = c.benchmark_group("random gen");
    groupe_random.sample_size(2_000);
    groupe_random.bench_function("creation time random SU3 adj", |b| {
        b.iter(|| create_su3_adj(&mut rng, &d))
    });
    groupe_random.finish();
}

criterion_group!(benches, criterion_benchmark, benchmark_base);
criterion_main!(benches);
