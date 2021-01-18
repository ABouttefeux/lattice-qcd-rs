
use super::{
    *,
    lattice::*,
    su3::*,
};
use std::vec::Vec;
use approx::*;

fn test_itrerator(points: usize){
    let l = LatticeCyclique::new(1_f64, points).unwrap();
    let array: Vec<LatticeLinkCanonical> = l.get_links(0).collect();
    assert_eq!(array.len(), 4 * points * points * points);
    let array: Vec<LatticePoint4> = l.get_points(0).collect();
    assert_eq!(array.len(), points * points * points);
}

#[test]
fn test_itrerator_length(){
    test_itrerator(2);
    test_itrerator(10);
    test_itrerator(26);
}

#[test]
fn test_exp(){
    let unit = 1_f64;
    assert_eq!(GENERATOR_1.exp(),CMatrix3::new(
        Complex::from(unit.cosh()),Complex::from(unit.sinh()), ZERO,
        Complex::from(unit.sinh()), Complex::from(unit.cosh()), ZERO,
        ZERO, ZERO, ONE
    ));
    assert_eq!(GENERATOR_2.exp(),CMatrix3::new(
        Complex::new(unit.cosh(), 0_f64), Complex::new(0_f64, - unit.sinh()), ZERO,
        Complex::new(0_f64, unit.sinh()), Complex::new(unit.cosh(), 0_f64), ZERO,
        ZERO, ZERO, ONE
    ));
}
