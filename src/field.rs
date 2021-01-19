
use super::{
    Real,
    CMatrix3,
    lattice::{
        LatticeLinkCanonical,
        LatticePoint,
        LatticeCyclique,
        PositiveF64,
    },
    Vector8,
    su3::{
        MatrixExp,
        GENERATORS,
    },
    I,
    ZERO,
    Complex,
    ONE,
};
use na::{
    Vector4,
    Matrix3
};
use  std::{
    collections::HashMap,
    //ops::{Deref, DerefMut},
};
//use t1ha::T1haHashMap;

type HashMapUse<K,V> = HashMap<K,V>;
type OutputNumber = Real;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Su3Adjoint {
    data: Vector8<OutputNumber>
}

impl Su3Adjoint {
    
    pub fn data(&self) -> &Vector8<OutputNumber> {
        &self.data
    }
    
    pub fn to_matrix(&self) -> Matrix3<na::Complex<OutputNumber>> {
        let mut mat = Matrix3::from_element(ZERO);
        for i in 0..self.data.len() {
            mat += *GENERATORS[i] * na::Complex::<OutputNumber>::from(self.data[i]);
        }
        return mat;
    }
    
    pub fn to_su3(self) -> Matrix3<na::Complex<OutputNumber>> {
        (self.to_matrix() * na::Complex::<OutputNumber>::i() ).exp()
    }
    
    pub fn random(rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<OutputNumber>) -> Self {
        Self {
            data : Vector8::<OutputNumber>::from_fn(|_,_| d.sample(rng))
        }
    }
    
}

#[derive(Debug)]
pub struct LinkMatrix {
    max_time : usize,
    data: HashMapUse<LatticeLinkCanonical, Matrix3<na::Complex<OutputNumber>>>,
}

impl LinkMatrix {
        
    pub fn data(&self) -> &HashMapUse<LatticeLinkCanonical, Matrix3<na::Complex<OutputNumber>>> {
        &self.data
    }
    
    pub fn new(l: &LatticeCyclique, rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<OutputNumber>) -> Self {
        let mut data = HashMapUse::default();
        data.reserve(l.get_number_of_canonical_links_space());
        for i in l.get_links_space(0) {
            let matrix = Su3Adjoint::random(rng, d).to_su3();
            data.insert(i, matrix);
        }
        Self {
            max_time: 0,
            data,
        }
    }
    
}

#[derive(Debug)]
pub struct EField
{
    max_time : usize,
    data: HashMapUse<LatticePoint, Vector4<Su3Adjoint>>,
}

impl EField {
    
    
    pub fn data(&self) -> &HashMapUse<LatticePoint, Vector4<Su3Adjoint>> {
        &self.data
    }
    
    pub fn new(l: &LatticeCyclique, rng: &mut rand::rngs::ThreadRng, d: &impl rand_distr::Distribution<OutputNumber>) -> Self {
        let mut data = HashMapUse::default();
        data.reserve(l.dim().pow(3));
        for i in l.get_points(0) {
            let p1 = Su3Adjoint::random(rng, d);
            let p2 = Su3Adjoint::random(rng, d);
            let p3 = Su3Adjoint::random(rng, d);
            let p4 = Su3Adjoint::random(rng, d);
            data.insert(i, Vector4::new(p1, p2, p3, p4));
        }
        Self {
            max_time: 0,
            data,
        }
    }
}

#[derive(Debug)]
pub struct LatticeSimulation {
    lattice : LatticeCyclique,
    e_field: EField,
    link_matrix: LinkMatrix,
}

impl LatticeSimulation {
    
    pub fn new(
        size: PositiveF64,
        number_of_points: usize,
        rng: &mut rand::rngs::ThreadRng,
        d: &impl rand_distr::Distribution<OutputNumber>,
    ) -> Option<Self> {
        let lattice_option = LatticeCyclique::new(size, number_of_points);
        if let None = lattice_option{
            return None;
        } 
        let lattice = lattice_option.unwrap();
        let e_field = EField::new(&lattice, rng, d);
        let link_matrix = LinkMatrix::new(&lattice, rng, d);
        Some(Self {
            lattice,
            e_field,
            link_matrix,
        })
    }
    
    pub fn e_field(&self) -> &EField {
        &self.e_field
    }
    
    pub fn link_matrix(&self) -> &LinkMatrix {
        &self.link_matrix
    }
    
    
    pub fn simulate(&mut self) {
        
    }
}
