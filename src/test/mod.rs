//! Module for testes

use std::{
    error::Error,
    f64::{self, consts::PI},
};

use approx::*;
use nalgebra::ComplexField;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::Uniform;

use super::{
    error::*, field::*, integrator::*, lattice::*, simulation::*, su3::*, thread::*, CMatrix3,
    Complex, Vector8, I, ONE, ZERO,
};

mod integrator;

/// Defines a small value to compare f64.
const EPSILON: f64 = 0.000_000_001_f64;

const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

/// test the size of iterators
fn test_iterator(points: usize) -> Result<(), LatticeInitializationError> {
    let l = LatticeCyclic::<4>::new(1_f64, points)?;
    let link_count = l.get_links().count();
    assert_eq!(link_count, 4 * points * points * points * points);
    assert_eq!(
        4 * points * points * points * points,
        l.number_of_canonical_links_space()
    );
    let pt_count = l.get_points().count();
    assert_eq!(pt_count, points * points * points * points);
    assert_eq!(pt_count, l.number_of_points());
    Ok(())
}

#[test]
/// test the size of iterators
fn test_iterator_length() -> Result<(), LatticeInitializationError> {
    test_iterator(2)?;
    test_iterator(10)?;
    test_iterator(26)
}

#[allow(deprecated)]
/// test the exponential of matrix
fn test_exp(factor: Complex) {
    let m_g1_exp = CMatrix3::new(
        factor.cosh(),
        factor.sinh(),
        ZERO,
        // ---
        factor.sinh(),
        factor.cosh(),
        ZERO,
        // ---
        ZERO,
        ZERO,
        ONE,
    );
    assert_eq_matrix!(
        (GENERATOR_1 * factor * Complex::from(2_f64)).exp(),
        m_g1_exp,
        EPSILON
    );
    let factor_i = factor * I;
    assert_eq_matrix!(
        (GENERATOR_2 * factor * Complex::from(2_f64)).exp(),
        CMatrix3::new(
            factor_i.cos(),
            -factor_i.sin(),
            ZERO,
            // ---
            factor_i.sin(),
            factor_i.cos(),
            ZERO,
            // ---
            ZERO,
            ZERO,
            ONE
        ),
        EPSILON
    );
}

#[test]
/// test the [`MatrixExp`] trait implementation
fn test_exp_old() {
    test_exp(ONE);
    test_exp(Complex::new(2_f64, 0_f64));
    test_exp(Complex::new(2_f64, 1_f64));
    test_exp(Complex::new(0_f64, 1_f64));
    test_exp(Complex::new(10.6_f64, 1_f64));
    test_exp(Complex::new(11.64_f64, -12.876_f64));
}

/// test [`su3_exp_i`] and [`Su3Adjoint::to_su3`]
#[allow(clippy::similar_names)] // factor_i and factor_mi are too similar. For this test it is not too confusing
fn test_exp_su3(factor: f64) {
    let factor_i = Complex::from(factor) * I;
    let m_g1_exp = CMatrix3::new(
        factor_i.cosh(),
        factor_i.sinh(),
        ZERO,
        // ---
        factor_i.sinh(),
        factor_i.cosh(),
        ZERO,
        // ---
        ZERO,
        ZERO,
        ONE,
    );
    let mut v1 = Vector8::zeros();
    v1[0] = factor * 2_f64;
    let gen1_equiv = Su3Adjoint::new(v1);
    assert_eq_matrix!(gen1_equiv.to_su3(), m_g1_exp, EPSILON);
    assert_eq_matrix!(su3_exp_i(gen1_equiv), m_g1_exp, EPSILON);

    let factor_mi = factor_i * I;
    let m_g2_exp = CMatrix3::new(
        factor_mi.cos(),
        -factor_mi.sin(),
        ZERO,
        // ---
        factor_mi.sin(),
        factor_mi.cos(),
        ZERO,
        // ---
        ZERO,
        ZERO,
        ONE,
    );

    let mut v2 = Vector8::zeros();
    v2[1] = factor * 2_f64;
    let gen2_equiv = Su3Adjoint::new(v2);
    assert_eq_matrix!(gen2_equiv.to_su3(), m_g2_exp, EPSILON);
    assert_eq_matrix!(su3_exp_i(gen2_equiv), m_g2_exp, EPSILON);
}

#[test]
/// basic test of [`su3_exp_i`] and [`Su3Adjoint::to_su3`]
fn test_exp_basic() {
    test_exp_su3(1_f64);
    test_exp_su3(2_f64);
    test_exp_su3(-1.254_f64);
    test_exp_su3(4.254_f64);
}

#[test]
#[allow(deprecated)]
/// test equivalence of [`su3_exp_i`] and [`MatrixExp`] trait implementation
fn equivalence_exp_i() {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let d = Uniform::from(-PI..PI);
    for i in 0..8 {
        let mut vec = Vector8::zeros();
        vec[i] = 1_f64;
        let v = Su3Adjoint::new(vec);
        let exp_r = su3_exp_i(v);
        let exp_l = (v.to_matrix() * Complex::new(0_f64, 1_f64)).exp();
        assert_eq_matrix!(exp_r, exp_l, EPSILON);
    }
    for _ in 0_u32..100_u32 {
        let v = Su3Adjoint::random(&mut rng, &d);
        let exp_r = su3_exp_i(v);
        let exp_l = (v.to_matrix() * Complex::new(0_f64, 1_f64)).exp();
        assert_eq_matrix!(exp_r, exp_l, EPSILON);
    }
}

#[test]
#[allow(deprecated)]
/// test equivalence of [`su3_exp_r`] and [`MatrixExp`] trait implementation
fn equivalence_exp_r() {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let d = Uniform::from(-PI..PI);
    for i in 0..8 {
        let mut vec = Vector8::zeros();
        vec[i] = 1_f64;
        let v = Su3Adjoint::new(vec);
        let exp_r = su3_exp_r(v);
        let exp_l = (v.to_matrix() * Complex::new(1_f64, 0_f64)).exp();
        assert_eq_matrix!(exp_r, exp_l, EPSILON);
    }
    for _ in 0_u32..100_u32 {
        let v = Su3Adjoint::random(&mut rng, &d);
        let exp_r = su3_exp_r(v);
        let exp_l = (v.to_matrix() * Complex::new(1_f64, 0_f64)).exp();
        assert_eq_matrix!(exp_r, exp_l, EPSILON);
    }
}

#[test]
/// test creation of sim ( single threaded)
fn create_sim() -> Result<(), ThreadedStateInitializationError> {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let distribution = Uniform::from(-PI..PI);
    let _simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(
        1_f64,
        1_f64,
        4,
        &mut rng,
        &distribution,
    )?;
    Ok(())
}

#[test]
/// test creation of sim multi threaded
fn create_sim_threaded() -> Result<(), ThreadedStateInitializationError> {
    let distribution = Uniform::from(-PI..PI);
    let _simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_random_threaded(
        1_f64,
        1_f64,
        4,
        &distribution,
        2,
    )?;
    Ok(())
}

/// return 1 if i==j 0 otherwise
const fn delta(i: usize, j: usize) -> f64 {
    if i == j {
        1_f64
    } else {
        0_f64
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::float_cmp)]
/// Test the properties of generators
fn test_generators() {
    for i in 0..7 {
        assert_eq!(GENERATORS[i].determinant(), ZERO);
    }
    for i in &GENERATORS {
        assert_eq!(i.trace(), ZERO);
        assert_eq!((i.adjoint() - **i).norm(), 0_f64);
    }
    for i in 0..GENERATORS.len() {
        for j in 0..GENERATORS.len() {
            assert_relative_eq!(
                (GENERATORS[i] * GENERATORS[j]).trace().modulus(),
                0.5_f64 * delta(i, j)
            );
        }
    }
}

#[test]
/// test the SU(3) properties of [`Su3Adjoint::to_su3`]
fn su3_property() {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let distribution = Uniform::from(-PI..PI);
    for _ in 0_u32..100_u32 {
        let m = Su3Adjoint::random(&mut rng, &distribution).to_su3();
        test_matrix_is_su3(&m);
    }
}

fn test_matrix_is_su3(m: &CMatrix3) {
    assert_matrix_is_su_3!(m, EPSILON);
}

#[test]
/// test [`run_pool_parallel`]
fn test_thread() -> Result<(), ThreadAnyError> {
    let c = 5_i32;
    let iter = 1_i32..10000_i32;
    for number_of_thread in &[1, 2, 4] {
        let result = run_pool_parallel(
            iter.clone(),
            &c,
            &|i, c| i * i * c,
            *number_of_thread,
            10000,
        )?;
        for i in iter.clone() {
            assert_eq!(result.get(&i).copied(), Some(i * i * c));
        }
    }
    Ok(())
}

#[test]
fn test_thread_error_zero_thread() {
    let iter = 1_i32..10000_i32;
    let result = run_pool_parallel(iter, &(), &|i, ()| i * i, 0, 10000);
    result.expect_err("zero thread request should give an error");
}

#[test]
/// test [`run_pool_parallel_vec`]
fn test_thread_vec() -> Result<(), Box<dyn Error>> {
    let l = LatticeCyclic::<4>::new(1_f64, 10)?;
    let iter = 0..10000;
    let c = 5;
    for number_of_thread in &[1, 2, 4] {
        let result = run_pool_parallel_vec(
            iter.clone(),
            &c,
            &|i, c| i * i * c,
            *number_of_thread,
            10000,
            &l,
            &0,
        )?;
        for i in iter.clone() {
            assert_eq!(result.get(i).copied(), Some(i * i * c));
        }
    }
    Ok(())
}

#[test]
fn test_thread_vec_error_zero_thread() -> Result<(), LatticeInitializationError> {
    let l = LatticeCyclic::<4>::new(1_f64, 10)?;
    let iter = 0..10000;
    let result = run_pool_parallel_vec(iter, &(), &|i, ()| i * i, 0, 10000, &l, &0);
    result.expect_err("zero thread should give an error");
    Ok(())
}

#[test]
/// test if Hamiltonian is more or less conserved over simulation
fn test_sim_hamiltonian() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let distribution = Uniform::from(-PI..PI);
    let simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(
        100_f64,
        1_f64,
        10,
        &mut rng,
        &distribution,
    )?;
    let h = simulation.hamiltonian_total();
    let sim2 = simulation.simulate_sync(&SymplecticEuler::new(8), 0.000_1_f64)?;
    let h2 = sim2.hamiltonian_total();
    println!("h1: {h}, h2: {h2}");
    assert!(h - h2 < 0.01_f64);
    Ok(())
}

#[test]
/// test if Gauss parameter is more or less conserved over simulation
fn test_gauss_law() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let distribution = Uniform::from(-PI..PI);
    let simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(
        1_f64,
        1_f64,
        10,
        &mut rng,
        &distribution,
    )?;
    let sim2 = simulation.simulate_sync(&SymplecticEuler::new(8), 0.000_001_f64)?;
    let iter_g_1 = simulation.lattice().get_points().map(|el| {
        simulation
            .gauss(&el)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)
    });
    let mut iter_g_2 = simulation.lattice().get_points().map(|el| {
        sim2.gauss(&el)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)
    });
    for g1 in iter_g_1 {
        let g2 = iter_g_2
            .next()
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(g1?, g2?, 0.001_f64);
    }
    Ok(())
}

#[test]
/// test if Hamiltonian is more or less conserved over simulation
fn test_sim_hamiltonian_rayon() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let distribution = Uniform::from(-PI..PI);
    let simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(
        100_f64,
        1_f64,
        10,
        &mut rng,
        &distribution,
    )?;
    let h = simulation.hamiltonian_total();
    let sim2 = simulation.simulate_sync(&SymplecticEulerRayon::new(), 0.000_1_f64)?;
    let h2 = sim2.hamiltonian_total();
    println!("h1: {h}, h2: {h2}");
    assert!(h - h2 < 0.01_f64);
    Ok(())
}

#[test]
/// test if Gauss parameter is more or less conserved over simulation
fn test_gauss_law_rayon() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    let distribution = Uniform::from(-PI..PI);
    let simulation = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_determinist(
        1_f64,
        1_f64,
        10,
        &mut rng,
        &distribution,
    )?;
    let sim2 = simulation.simulate_sync(&SymplecticEulerRayon::new(), 0.000_001_f64)?;
    let iter_g_1 = simulation.lattice().get_points().map(|el| {
        simulation
            .gauss(&el)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)
    });
    let mut iter_g_2 = simulation.lattice().get_points().map(|el| {
        sim2.gauss(&el)
            .ok_or(ImplementationError::OptionWithUnexpectedNone)
    });
    for g1 in iter_g_1 {
        let g2 = iter_g_2
            .next()
            .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
        assert_eq_matrix!(g1?, g2?, 0.001_f64);
    }
    Ok(())
}

#[test]
/// test that the simulation of a cold state does not change over time
fn test_sim_cold() -> Result<(), Box<dyn Error>> {
    let size = 10_f64;
    let number_of_pts = 10;
    let beta = 0.1_f64;
    let sim1 = LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_cold(
        size,
        beta,
        number_of_pts,
    )?;
    let sim2 = sim1.simulate_to_leapfrog(&SymplecticEulerRayon::new(), 0.1_f64)?;
    assert_eq!(sim1.e_field(), sim2.e_field());
    assert_eq!(sim1.link_matrix(), sim2.link_matrix());
    let sim3 = sim2.simulate_leap(&SymplecticEulerRayon::new(), 0.1_f64)?;
    assert_eq!(sim2.e_field(), sim3.e_field());
    assert_eq!(sim2.link_matrix(), sim3.link_matrix());
    let sim4 = sim3.simulate_to_synchronous(&SymplecticEulerRayon::new(), 0.1_f64)?;
    assert_eq!(sim3.e_field(), sim4.e_field());
    assert_eq!(sim3.link_matrix(), sim4.link_matrix());
    Ok(())
}

// cspell: ignore orthonormalization
#[allow(clippy::too_many_lines)]
#[test]
fn orthonormalization() {
    assert_eq!(orthonormalize_matrix(&CMatrix3::zeros()), CMatrix3::zeros());
    assert_eq!(
        orthonormalize_matrix(&CMatrix3::identity()),
        CMatrix3::identity()
    );

    let m = nalgebra::Matrix3::<Complex>::new(
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
    );
    assert_eq!(orthonormalize_matrix(&m), m);
    let m = nalgebra::Matrix3::<Complex>::new(
        Complex::new(2_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(2_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
    );
    assert_eq!(orthonormalize_matrix(&m), CMatrix3::identity());
    let m = nalgebra::Matrix3::<Complex>::new(
        Complex::new(0_f64, 2_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 2_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
    );
    assert_eq!(
        orthonormalize_matrix(&m),
        CMatrix3::from_diagonal(&nalgebra::Vector3::new(
            Complex::i(),
            Complex::i(),
            Complex::new(-1_f64, 0_f64)
        ))
    );

    let m = nalgebra::Matrix3::<Complex>::new(
        Complex::new(0_f64, 0_f64),
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
    );
    let mc = nalgebra::Matrix3::<Complex>::new(
        Complex::new(0_f64, 0_f64),
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(-1_f64, 0_f64),
    );
    assert_eq!(orthonormalize_matrix(&m), mc);

    let m = nalgebra::Matrix3::<Complex>::new(
        Complex::new(0_f64, 0_f64),
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
    );
    let mc = nalgebra::Matrix3::<Complex>::new(
        Complex::new(0_f64, 0_f64),
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        // ---
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(1_f64, 0_f64),
        // ---
        Complex::new(1_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
        Complex::new(0_f64, 0_f64),
    );
    assert_eq!(orthonormalize_matrix(&m), mc);

    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    //let d = Uniform::from(-10_f64..10_f64);
    for _ in 0_u32..100_u32 {
        let m = random_matrix_3(&mut rng);
        println!("{} , {}", m, orthonormalize_matrix(&m));
        if m.determinant() == Complex::from(0_f64) {
            assert_eq!(
                orthonormalize_matrix(&m).determinant(),
                Complex::from(0_f64)
            );
        } else {
            test_matrix_is_su3(&orthonormalize_matrix(&m));
        }
    }

    for _ in 0_u32..100_u32 {
        let mut m = random_matrix_3(&mut rng);
        if m.determinant() != Complex::from(0_f64) {
            orthonormalize_matrix_mut(&mut m);
            test_matrix_is_su3(&m);
        }
    }
}

#[test]
fn random_su3_t() {
    let mut rng = StdRng::seed_from_u64(SEED_RNG);
    for _ in 0_u32..100_u32 {
        test_matrix_is_su3(&random_su3(&mut rng));
    }
}

#[test]
fn lattice_init_error() -> Result<(), LatticeInitializationError> {
    assert_eq!(
        LatticeCyclic::<3>::new(0_f64, 4),
        Err(LatticeInitializationError::NonPositiveSize)
    );
    assert_eq!(
        LatticeCyclic::<4>::new(-1_f64, 4),
        Err(LatticeInitializationError::NonPositiveSize)
    );
    assert_eq!(
        LatticeCyclic::<4>::new(f64::NAN, 4),
        Err(LatticeInitializationError::NonPositiveSize)
    );
    assert_eq!(
        LatticeCyclic::<4>::new(-0_f64, 4),
        Err(LatticeInitializationError::NonPositiveSize)
    );
    assert_eq!(
        LatticeCyclic::<4>::new(f64::INFINITY, 4),
        Err(LatticeInitializationError::NonPositiveSize)
    );
    assert_eq!(
        LatticeCyclic::<3>::new(1_f64, 1),
        Err(LatticeInitializationError::DimTooSmall)
    );
    assert_eq!(
        LatticeCyclic::<3>::new(1_f64, 1),
        Err(LatticeInitializationError::DimTooSmall)
    );
    LatticeCyclic::<3>::new(1_f64, 2)?;
    assert_eq!(
        LatticeCyclic::<0>::new(1_f64, 2),
        Err(LatticeInitializationError::ZeroDimension)
    );
    Ok(())
}

#[test]
fn state_init_error() {
    let d = Uniform::from(-PI..PI);

    assert_eq!(
        LatticeStateEFSyncDefault::<LatticeStateDefault<3>, 3>::new_random_threaded(
            0_f64, 1_f64, 4, &d, 2,
        ),
        Err(ThreadedStateInitializationError::StateInitializationError(
            StateInitializationError::LatticeInitializationError(
                LatticeInitializationError::NonPositiveSize
            )
        ))
    );

    assert_eq!(
        LatticeStateEFSyncDefault::<LatticeStateDefault<3>, 3>::new_random_threaded(
            1_f64, 1_f64, 4, &d, 0,
        ),
        Err(ThreadedStateInitializationError::ThreadingError(
            ThreadError::ThreadNumberIncorrect
        ))
    );
}

#[test]
fn test_length_compatible() -> Result<(), LatticeInitializationError> {
    let l1 = LatticeCyclic::<2>::new(1_f64, 4)?;
    let l2 = LatticeCyclic::<2>::new(1_f64, 3)?;
    let link = LinkMatrix::new_cold(&l1);
    let e_f = EField::new_cold(&l1);
    assert!(l1.has_compatible_length(&link, &e_f));
    let link2 = LinkMatrix::new_cold(&l2);
    let e_f_2 = EField::new_cold(&l2);
    assert!(!l1.has_compatible_length(&link2, &e_f));
    assert!(!l1.has_compatible_length(&link2, &e_f_2));
    assert!(l2.has_compatible_length(&link2, &e_f_2));
    Ok(())
}

#[test]
fn test_leap_frog() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(0);
    let size = 1000_f64;
    let number_of_pts = 4;
    let beta = 1_f64;
    let state = LatticeStateDefault::new_determinist(size, beta, number_of_pts, &mut rng)?;
    println!("h_l {}", state.hamiltonian_links());
    let state_hmc =
        LatticeStateEFSyncDefault::<LatticeStateDefault<4>, 4>::new_random_e_state(state, &mut rng);
    let h1 = state_hmc.hamiltonian_total();
    println!("h_t {h1}");
    let state_hmc_2 =
        state_hmc.simulate_using_leapfrog_n_auto(&SymplecticEulerRayon::new(), 0.01_f64, 1)?;
    let h2 = state_hmc_2.hamiltonian_total();
    println!("h_t {h2}");
    println!("{}", (h1 - h2).exp());

    assert!((h1 - h2).abs() < 0.000_01_f64);
    Ok(())
}
