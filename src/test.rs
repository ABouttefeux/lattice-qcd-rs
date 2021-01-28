
//! Module for testes

use super::{
    lattice::*,
    su3::*,
    field::*,
    ONE,
    CMatrix3,
    Complex,
    ZERO,
    I,
    Vector8,
    thread::*,
    integrator::*,
};
use std::{
    f64,
    vec::Vec,
};
use approx::*;
use na::ComplexField;

/// Defines a small value to compare f64.
const EPSILON: f64 = 0.000000001_f64;

/// test the size of iterators
fn test_itrerator(points: usize){
    let l = LatticeCyclique::new(1_f64, points).unwrap();
    let array: Vec<LatticeLinkCanonical> = l.get_links_space(0).collect();
    assert_eq!(array.len(), 3 * points * points * points);
    assert_eq!(3 * points * points * points, l.get_number_of_canonical_links_space());
    let array: Vec<LatticePoint> = l.get_points(0).collect();
    assert_eq!(array.len(), points * points * points);
    assert_eq!(array.len(), l.get_number_of_points());
}

#[test]
/// test the size of iterators
fn test_itrerator_length(){
    test_itrerator(2);
    test_itrerator(10);
    test_itrerator(26);
}

/// test the exponential of matrix
fn test_exp(factor : Complex){
    let m_g1_exp = CMatrix3::new(
        factor.cosh(), factor.sinh(), ZERO,
        factor.sinh(), factor.cosh(), ZERO,
        ZERO, ZERO, ONE
    );
    assert!(((*GENERATOR_1 * factor * Complex::from(2_f64)).exp() - m_g1_exp).norm() < EPSILON);
    let factor_i = factor * I;
    assert!(((*GENERATOR_2 * factor * Complex::from(2_f64)).exp() - CMatrix3::new(
        factor_i.cos(), - factor_i.sin(), ZERO,
        factor_i.sin(), factor_i.cos(), ZERO,
        ZERO, ZERO, ONE
    )).norm() < EPSILON);
}

#[test]
/// test the [`MatrixExp`] trait implementation
fn test_exp_old(){
    test_exp(ONE);
    test_exp(Complex::new(2_f64, 0_f64));
    test_exp(Complex::new(2_f64, 1_f64));
    test_exp(Complex::new(0_f64, 1_f64));
    test_exp(Complex::new(10.6_f64, 1_f64));
    test_exp(Complex::new(11.64_f64, -12.876_f64));
}

/// test [`su3_exp_i`] and [`Su3Adjoint::to_su3`]
fn test_exp_su3(factor: f64){
    let factor_i = Complex::from(factor) * I;
    let m_g1_exp = CMatrix3::new(
        factor_i.cosh(), factor_i.sinh(), ZERO,
        factor_i.sinh(), factor_i.cosh(), ZERO,
        ZERO, ZERO, ONE
    );
    let mut v1 = Vector8::zeros();
    v1[0] = factor * 2_f64;
    let gen1_equiv = Su3Adjoint::new(v1);
    assert!( (gen1_equiv.to_su3() - m_g1_exp).norm() < EPSILON);
    assert!( (su3_exp_i(gen1_equiv) - m_g1_exp).norm() < EPSILON);
    
    let factor_mi = factor_i * I;
    let m_g2_exp = CMatrix3::new(
        factor_mi.cos(), - factor_mi.sin(), ZERO,
        factor_mi.sin(), factor_mi.cos(), ZERO,
        ZERO, ZERO, ONE
    );
    
    let mut v2 = Vector8::zeros();
    v2[1] = factor * 2_f64;
    let gen2_equiv = Su3Adjoint::new(v2);
    assert!( (gen2_equiv.to_su3() - m_g2_exp).norm() < EPSILON);
    assert!( (su3_exp_i(gen2_equiv) - m_g2_exp).norm() < EPSILON);
}

#[test]
/// basic test of [`su3_exp_i`] and [`Su3Adjoint::to_su3`]
fn test_exp_basic(){
    test_exp_su3(1_f64);
    test_exp_su3(2_f64);
    test_exp_su3(-1.254_f64);
}

#[test]
/// test equivalence of [`su3_exp_i`] and [`MatrixExp`] trait implementation
fn equivalece_exp_i(){
    let mut rng = rand::thread_rng();
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    for i in 0..8 {
        let mut vec = Vector8::zeros();
        vec[i] = 1_f64;
        let v = Su3Adjoint::new(vec);
        let exp_r = su3_exp_i(v);
        let exp_l = (v.to_matrix() * Complex::new(0_f64, 1_f64)).exp();
        assert!(( exp_r - exp_l).norm() < EPSILON );
    }
    for _ in 0..100 {
        let v = Su3Adjoint::random(&mut rng, &d);
        let exp_r = su3_exp_i(v);
        let exp_l = (v.to_matrix() * Complex::new(0_f64, 1_f64)).exp();
        assert!(( exp_r - exp_l).norm() < EPSILON );
    }
}

#[test]
/// test equivalence of [`su3_exp_r`] and [`MatrixExp`] trait implementation
fn equivalece_exp_r(){
    let mut rng = rand::thread_rng();
    let d = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    for i in 0..8 {
        let mut vec = Vector8::zeros();
        vec[i] = 1_f64;
        let v = Su3Adjoint::new(vec);
        let exp_r = su3_exp_r(v);
        let exp_l = (v.to_matrix() * Complex::new(1_f64, 0_f64)).exp();
        assert!(( exp_r - exp_l).norm() < EPSILON );
    }
    for _ in 0..100 {
        let v = Su3Adjoint::random(&mut rng, &d);
        let exp_r = su3_exp_r(v);
        let exp_l = (v.to_matrix() * Complex::new(1_f64, 0_f64)).exp();
        assert!(( exp_r - exp_l).norm() < EPSILON );
    }
}

#[test]
/// test creation of sim ( single threaded)
fn create_sim() {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let _simulation = LatticeSimulationState::new_deterministe(1_f64 , 4, &mut rng, &distribution).unwrap();
}

#[test]
/// test creation of sim multi threaded
fn creat_sim_threaded() {
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let _simulation = LatticeSimulationState::new_random_threaded(1_f64, 4, &distribution, 2).unwrap();
}

/// return 1 if i==j 0 otherwise
fn delta(i: usize, j: usize) -> f64{
    if i == j {
        return 1_f64;
    }
    else {
        return 0_f64;
    }
}

#[test]
/// Test the properties of generators
fn test_generators() {
    for i in 0..7{
        assert_eq!( GENERATORS[i].determinant(), ZERO);
    }
    for i in &*GENERATORS {
        assert_eq!( i.trace(), ZERO);
        assert_eq!( (i.adjoint() - **i).norm(), 0_f64);
    }
    for i in 0..GENERATORS.len(){
        for j in 0..GENERATORS.len(){
            assert_relative_eq!((GENERATORS[i] * GENERATORS[j]).trace().modulus(), 0.5_f64 * delta(i,j));
        }
    }
}

#[test]
/// test the SU(3) properties of [`Su3Adjoint::to_su3`]
fn test_su3_property(){
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    for _ in 0..100 {
        let m = Su3Adjoint::random(&mut rng, &distribution).to_su3();
        assert!((m.determinant().modulus_squared() - 1_f64).abs() < EPSILON);
        assert!((m * m.adjoint() - CMatrix3::identity()).norm() < EPSILON);
    }
}

#[test]
/// test [`run_pool_parallel`]
fn test_thread() {
    let iter = 1..10000;
    let c = 5;
    let result = run_pool_parallel(iter.clone(), &c, &|i, c| {i * i * c} , 4, 10000).unwrap();
    for i in iter {
        assert_eq!(* result.get(&i).unwrap(), i * i *c);
    }
}

#[test]
/// test [`run_pool_parallel_vec`]
fn test_thread_vec() {
    let l = LatticeCyclique::new(1_f64, 10).unwrap();
    let iter = 0..10000;
    let c = 5;
    let result = run_pool_parallel_vec(iter.clone(), &c, &|i, c| {i * i * c} , 4, 10000, &l, 0).unwrap();
    for i in iter {
        assert_eq!(*result.get(i).unwrap(), i * i *c);
    }
}

#[test]
/// test if Hamiltonian is more or less conserved over simulation
fn test_sim_hamiltonian() {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let simulation = LatticeSimulationState::new_deterministe(100_f64 , 20, &mut rng, &distribution).unwrap();
    let h = simulation.get_hamiltonian();
    let sim2 = simulation.simulate::<SymplecticEuler>(0.0001, 8).unwrap();
    let h2 = sim2.get_hamiltonian();
    println!("h1: {}, h2: {}", h, h2);
    assert!(h - h2 < 0.01_f64 );
}

#[test]
/// test if Gauss parameter is more or less conserved over simulation
fn test_gauss_law() {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let simulation = LatticeSimulationState::new_deterministe(1_f64 , 20, &mut rng, &distribution).unwrap();
    let sim2 = simulation.simulate::<SymplecticEuler>(0.000001, 8).unwrap();
    let iter_g_1 = simulation.lattice().get_points(0).map(|el| {
        simulation.get_gauss(&el).unwrap()
    });
    let mut iter_g_2 = simulation.lattice().get_points(0).map(|el| {
        sim2.get_gauss(&el).unwrap()
    });
    for g1 in iter_g_1 {
        let g2 = iter_g_2.next().unwrap();
        assert!((g1 - g2).norm() < 0.001);
    }
}
