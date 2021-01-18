
use super::{
    Real,
    lattice::{
        LatticeLinkCanonical,
        LatticePoint4,
        LatticeCyclique,
    },
};
use na::{
    Vector4,
};
use  std::{
    collections::HashMap,
};

struct LinkMatrix {
    data: HashMap<LatticeLinkCanonical, Real>,
}

impl LinkMatrix{
    
    pub fn data(&self) -> &HashMap<LatticeLinkCanonical, Real> {
        &self.data
    }
    
    pub fn new(l: &LatticeCyclique, min: Real, max: Real) -> Self {
        Self{data : HashMap::new()}
    }
    
}

struct EField {
    data: HashMap<LatticePoint4, Vector4<Real>>,
}

impl EField {
    
    pub fn data(&self) -> &HashMap<LatticePoint4, Vector4<Real>> {
        &self.data
    }
    
}

pub struct LatticeSimulation {
    lattice : LatticeCyclique,
    e_feild: EField,
    link_matrix: LinkMatrix,
}

impl LatticeSimulation {

}
