
use na::{
    ComplexField,
    MatrixN,
    base::allocator::Allocator,
    DefaultAllocator,
};
use std::{
     convert::{From, Into},
};
use super::{
    Complex,
    ONE,
    I,
    ZERO,
    CMatrix3,
};
use once_cell::sync::Lazy;

pub static GENERATOR_1: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ONE, ZERO,
        ONE, ZERO, ZERO,
        ZERO, ZERO, ZERO,
));

pub static GENERATOR_2: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, -I, ZERO,
        I, ZERO, ZERO,
        ZERO, ZERO, ZERO,
));

pub static GENERATOR_3: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ONE, ZERO, ZERO,
        ZERO, -ONE, ZERO,
        ZERO, ZERO, ZERO,
));

pub static GENERATOR_4: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, ONE,
        ZERO, ZERO, ZERO,
        ONE, ZERO, ZERO,
));

pub static GENERATOR_5: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, -I,
        ZERO, ZERO, ZERO,
        I, ZERO, ZERO,
));

pub static GENERATOR_6: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, ZERO,
        ZERO, ZERO, ONE,
        ZERO, ONE, ZERO,
));

pub static GENERATOR_7: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        ZERO, ZERO, ZERO,
        ZERO, ZERO, -I,
        ZERO, I, ZERO,
));

pub static GENERATOR_8: Lazy<CMatrix3> = Lazy::new(|| CMatrix3::new(
        Complex::new(1_f64 / 3_f64.sqrt(), 0_f64), ZERO, ZERO,
        ZERO, Complex::new(1_f64 / 3_f64.sqrt(), 0_f64), ZERO,
        ZERO, ZERO, Complex::new(-2_f64 / 3_f64.sqrt(), 0_f64),
));

pub static GENERATORS: Lazy<[&CMatrix3; 8]> = Lazy::new(|| 
    [&GENERATOR_1, &GENERATOR_2, &GENERATOR_3, &GENERATOR_4, &GENERATOR_5, &GENERATOR_6, &GENERATOR_7, &GENERATOR_1]
);


pub trait MatrixExp {
    fn exp(&self) -> Self;
}


impl<T, D> MatrixExp for MatrixN<T, D>
    where T: ComplexField + Copy,
    D: na::DimName + na::DimSub<na::U1>,
    DefaultAllocator: Allocator<T, D, na::DimDiff<D, na::U1>>
        + Allocator<T, na::DimDiff<D, na::U1>>
        + Allocator<T, D, D>
        + Allocator<T, D>,
    MatrixN<T, D> : Clone,
{
    fn exp(&self) -> Self {
        let decomposition = self.clone().schur();
        // a complex matrix is always diagonalisable
        let eigens = decomposition.eigenvalues().unwrap();
        let mut new_matrix = Self::identity();
        for i in 0..new_matrix.nrows() {
            new_matrix[(i, i)] = eigens[i].exp();
        }
        let (q, _) = decomposition.unpack();
        // q is always invertible
        return q.clone() * new_matrix * q.try_inverse().unwrap();
    }
}
