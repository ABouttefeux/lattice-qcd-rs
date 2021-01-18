
extern crate nalgebra as na;
extern crate once_cell;

use std::ops::Mul;


use na::{
    ArrayStorage,
    Matrix,
    VectorN,
    Matrix3,
    ComplexField,
};
use std::{
     convert::{From, Into},
};
use super::{
    Complex,
    ONE,
    I,
    ZERO,
};
use once_cell::sync::Lazy;


pub type Vector8<N> = VectorN<N, na::U8>;



pub type CMatrix3 = Matrix3<Complex>;

pub static GENERATOR_1: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ZERO, ONE, ZERO,
        ONE, ZERO, ZERO,
        ZERO, ZERO, ZERO,
));

pub static GENERATOR_2: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ZERO, -I, ZERO,
        I, ZERO, ZERO,
        ZERO, ZERO, ZERO,
));

pub static GENERATOR_3: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ONE, ZERO, ZERO,
        ZERO, -ONE, ZERO,
        ZERO, ZERO, ZERO,
));

pub static GENERATOR_4: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ZERO, ZERO, ONE,
        ZERO, ZERO, ZERO,
        ONE, ZERO, ZERO,
));

pub static GENERATOR_5: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ZERO, ZERO, I,
        ZERO, ZERO, ZERO,
        I, ZERO, ZERO,
));

pub static GENERATOR_6: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ZERO, ZERO, ZERO,
        ZERO, ZERO, ONE,
        ZERO, ONE, ZERO,
));

pub static GENERATOR_7: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        ZERO, ZERO, ZERO,
        ZERO, ZERO, -I,
        ZERO, -I, ZERO,
));

pub static GENERATOR_8: Lazy<CMatrix3> = Lazy::new(|| Matrix3::new(
        Complex::new(1_f64 / 3_f64.sqrt(), 0_f64), ZERO, ZERO,
        ZERO, Complex::new(1_f64 / 3_f64.sqrt(), 0_f64), ZERO,
        ZERO, ZERO, Complex::new(-2_f64 / 3_f64.sqrt(), 0_f64),
));

pub static GENERATORS: Lazy<[&CMatrix3; 8]> = Lazy::new(|| 
    [&GENERATOR_1, &GENERATOR_2, &GENERATOR_3, &GENERATOR_4, &GENERATOR_5, &GENERATOR_6, &GENERATOR_7, &GENERATOR_1]
);


pub trait Exp {
    fn exp(&self) -> Self;
}

impl Exp for CMatrix3 {
    
    fn exp(&self) -> Self {
        let decomposition = self.schur();
        // a complex matrix is always diagonalisable
        let eigens = decomposition.eigenvalues().unwrap();
        let mut new_matrix = CMatrix3::identity();
        for i in 0..3 {
            new_matrix[(i, i)] = eigens[i].exp();
        }
        let (q, _) = decomposition.unpack();
        // q is always invertible
        return q * new_matrix * q.try_inverse().unwrap();
    }
}
