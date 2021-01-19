
use super::{
    lattice::*,
    su3::*,
    field::*,
    ONE,
    CMatrix3,
    Complex,
    ZERO,
    I,
};
use std::{
    f64,
    vec::Vec,
};
use approx::*;
use na::ComplexField;

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
fn test_itrerator_length(){
    test_itrerator(2);
    test_itrerator(10);
    test_itrerator(26);
}

const EPSILON: f64 = 0.00001_f64;

fn test_exp(factor : Complex){
    assert!(((*GENERATOR_1 * factor).exp() - CMatrix3::new(
        factor.cosh(), factor.sinh(), ZERO,
        factor.sinh(), factor.cosh(), ZERO,
        ZERO, ZERO, ONE
    )).norm() < EPSILON);
    let factor_i = factor * I;
    assert!(((*GENERATOR_2 * factor).exp() - CMatrix3::new(
        factor_i.cos(), - factor_i.sin(), ZERO,
        factor_i.sin(), factor_i.cos(), ZERO,
        ZERO, ZERO, ONE
    )).norm() < EPSILON);
}

#[test]
fn test_exp_mul(){
    test_exp(ONE);
    test_exp(Complex::new(2_f64, 0_f64));
    test_exp(Complex::new(2_f64, 1_f64));
    test_exp(Complex::new(0_f64, 1_f64));
    test_exp(Complex::new(10.6_f64, 1_f64));
    test_exp(Complex::new(11.64_f64, -12.876_f64));
}

#[test]
fn create_sim() {
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    let mut simulation = LatticeSimulation::new(1_f64 , 4, &mut rng, &distribution).unwrap();
}

#[test]
fn test_generators() {
    for i in &*GENERATORS {
        assert_eq!( i.trace(), ZERO);
        assert_eq!( i.determinant(), ZERO);
        assert_eq!( (i.adjoint() - **i).norm(), 0_f64);
    }
}

#[test] 
fn test_su3(){
    let mut rng = rand::thread_rng();
    let distribution = rand::distributions::Uniform::from(-f64::consts::PI..f64::consts::PI);
    for _i in 0..100 {
        let m = Su3Adjoint::random(&mut rng, &distribution).to_su3();
        assert!((m.determinant().modulus_squared() - 1_f64).abs() < EPSILON);
        assert!((m.adjoint() * m - CMatrix3::identity()).norm() < EPSILON);
    }
}
