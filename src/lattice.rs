
use na::{
    Vector4
};
use approx::*;
use super::Real;

pub type PositiveF64 = Real;

#[derive(Clone, Debug)]
pub struct LatticeCyclique {
    size: PositiveF64,
    dim: usize,
}

impl LatticeCyclique {
    
    pub const DIM: usize = 4;
    
    pub fn get_link_canonical(&self, pos: &[usize; LatticeCyclique::DIM], dir: &Direction) -> LatticeLinkCanonical {
        let mut pos_link: [usize; LatticeCyclique::DIM] = [0; LatticeCyclique::DIM];
        if dir.is_positive() {
            for i in 0..pos.len() {
                pos_link[i] = pos[i] % self.dim();
            }
            return LatticeLinkCanonical::new(pos_link, dir.clone()).unwrap();
        }
        else {
            let vec = dir.to_vector(1_f64);
            for i in 0..pos.len() {
                let diff = - vec[i] as usize;
                if pos[i] == 0 && diff == 1 {
                    pos_link[i] = self.dim - 1_usize;
                }
                else {
                    pos_link[i] = pos[i] - diff;
                }
                pos_link[i] = pos_link[i] % self.dim();
            }
            return LatticeLinkCanonical::new(pos_link, dir.to_positive()).unwrap();
        }
    }
    
    pub fn get_link (&self, pos: &[usize; LatticeCyclique::DIM], dir: &Direction) -> LatticeLink {
        let mut pos_link = [0_usize; LatticeCyclique::DIM];
        for i in 0..pos.len() {
            pos_link[i] = pos[i] % self.dim();
        }
        LatticeLink::new(pos_link, dir.clone())
    }
    
    pub fn get_canonical(&self, l: &LatticeLink) -> LatticeLinkCanonical{
        self.get_link_canonical(l.pos(), l.dir())
    }
    
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    // TODO iterators ?
    pub fn get_links_space(&self, time_pos: usize) -> IteratorLatticeLinkCanonical {
        return IteratorLatticeLinkCanonical::new(&self, &self.get_link_canonical(&[0, 0, 0, time_pos], Direction::POSITIVES_SPACE.first().unwrap()));
    }
    
    pub fn get_points(&self, time_pos: usize) ->  IteratorLatticePoint{
        return IteratorLatticePoint::new(&self, &[0, 0, 0, time_pos]);
    }
    
    pub fn new(size: PositiveF64, dim: usize) -> Option<Self>{
        if size < 0_f64 {
            return None;
        }
        if dim < 2 {
            return None;
        }
        return Some(Self {
            size, dim
        })
    }
    
    pub fn get_number_of_canonical_links_space(&self) -> usize {
        self.get_number_of_points() * 3
    }
    
    pub fn get_number_of_points(&self) -> usize {
        self.dim().pow(3)
    }
    
}

#[derive(Clone, Debug)]
pub struct IteratorLatticeLinkCanonical<'a> {
    lattice: &'a LatticeCyclique,
    element: Option<LatticeLinkCanonical>,
}

impl<'a> IteratorLatticeLinkCanonical<'a> {
    fn new(lattice: &'a LatticeCyclique, first_el: &LatticeLinkCanonical) -> Self {
        Self {
            lattice,
            element: Some(first_el.clone()),
        }
    }
}


impl<'a> Iterator for IteratorLatticeLinkCanonical<'a> {
    type Item = LatticeLinkCanonical;
    
    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        const LINK_ARRAY: [Direction; 3] = Direction::POSITIVES_SPACE;
        let previous_el = self.element.clone();
        match &mut self.element {
            Some(element) => {
                let mut iter_dir = LINK_ARRAY.iter();
                iter_dir.find(|el| *el == element.dir());
                let new_dir = iter_dir.next();
                match new_dir {
                    Some(dir) => *element.dir_mut() = dir.clone(),
                    None => {
                        *element.dir_mut() = LINK_ARRAY.first().unwrap().clone();
                        let mut iter = IteratorLatticePoint::new(self.lattice, element.pos());
                        iter.next();
                        match iter.next() {
                            Some(array) => *element.pos_mut() = array,
                            None => {
                                self.element = None;
                                return previous_el;
                            },
                        }
                    }
                }
            },
            None => (),
        }
        
        return previous_el;
    }
}

#[derive(Clone, Debug)]
pub struct IteratorLatticePoint<'a>  {
    lattice: &'a LatticeCyclique,
    element: Option<LatticePoint>,
}

impl<'a> IteratorLatticePoint<'a> {
    fn new(lattice: &'a LatticeCyclique, first_el: &LatticePoint) -> Self {
        Self {
            lattice,
            element: Some(first_el.clone()),
        }
    }
}


impl<'a> Iterator for IteratorLatticePoint<'a> {
    type Item = LatticePoint;
    
    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        let previous_el = self.element.clone();
        match &mut self.element {
            Some(el) => {
                el[0] += 1;
                for i in 0..el.len() - 1 {
                    while el[i] >= self.lattice.dim() {
                        if i <= 1 {
                            el[i + 1] += 1;
                        }   
                        else{
                            self.element = None;
                            return previous_el;
                        }
                        el[i] -= self.lattice.dim()
                    }
                }
            },
            None => (),
        }
        return previous_el;
    }
}


pub type LatticePoint = [usize; 4];

impl From<LatticeLink> for LatticePoint {
    fn from(f: LatticeLink) -> Self{
        f.from
    }
} 

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LatticeLinkCanonical {
    from: [usize; 4],
    dir: Direction,
}

impl LatticeLinkCanonical {
    pub fn new (from: [usize;4], dir: Direction) -> Option<Self> {
        if dir.is_negative() {
            return None;
        }
        Some(Self {from, dir})
    }
    
    pub fn pos(&self) -> &[usize; 4]{
        &self.from
    }
    
    pub fn pos_mut(&mut self) -> &mut [usize; 4]{
        &mut self.from
    }
    
    pub fn dir(&self) -> &Direction {
        &self.dir
    }
    
    pub fn dir_mut(&mut self) -> &mut Direction {
        &mut self.dir
    }
}

impl From<LatticeLinkCanonical> for LatticeLink {
    fn from(l : LatticeLinkCanonical) -> Self {
        LatticeLink::new(l.from, l.dir)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LatticeLink {
    from: [usize; 4],
    dir: Direction,
}

impl LatticeLink {
    pub fn new (from: [usize;4], dir: Direction) -> Self {
        Self {from, dir}
    }
    
    pub fn pos(&self) -> &[usize; 4]{
        &self.from
    }
    
    pub fn pos_mut(&mut self) -> &mut [usize; 4]{
        &mut self.from
    }
    
    pub fn dir(&self) -> &Direction {
        &self.dir
    }
    
    pub fn dir_mut(&mut self) -> &mut Direction {
        &mut self.dir
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Sign {
    Negative, Positive, Zero
}

impl Sign {
    pub fn to_f64(&self) -> f64 {
        match self {
            Sign::Negative => -1_f64,
            Sign::Positive => 1_f64,
            Sign::Zero => 0_f64,
        }
    }
    
    pub fn sign(f: Real) -> Self {
        if relative_eq!(f, 0_f64) {
            return Sign::Zero;
        }
        if f > 0_f64{
            return Sign::Positive;
        }
        else {
            return Sign::Negative;
        }
    }
    
}

impl From<Sign> for f64 {
    fn from(s : Sign) -> f64 {
        s.to_f64()
    }
}

impl From<f64> for Sign {
    fn from(f : f64) -> Sign {
        Sign::sign(f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    XPos, XNeg, YPos, YNeg, ZPos, ZNeg, TPos, TNeg,
}

impl Direction{
    pub fn to_vector(&self, a: f64) -> Vector4<Real> {
        match self {
            Direction::XPos => Vector4::<Real>::new(1_f64 * a, 0_f64, 0_f64, 0_f64),
            Direction::XNeg => Vector4::<Real>::new(-1_f64 * a, 0_f64, 0_f64, 0_f64),
            Direction::YPos => Vector4::<Real>::new(0_f64, 1_f64 * a, 0_f64, 0_f64),
            Direction::YNeg => Vector4::<Real>::new(0_f64, -1_f64 * a, 0_f64, 0_f64),
            Direction::ZPos => Vector4::<Real>::new(0_f64, 0_f64, 1_f64 * a, 0_f64),
            Direction::ZNeg => Vector4::<Real>::new(0_f64, 0_f64, -1_f64 * a, 0_f64),
            Direction::TPos => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, 1_f64 * a),
            Direction::TNeg => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, -1_f64 * a),
        }
    }
    

    pub fn is_positive(&self) -> bool {
        match self {
            Direction::XPos | Direction::YPos | Direction::ZPos | Direction::TPos => true,
            Direction::XNeg | Direction::YNeg | Direction::ZNeg | Direction::TNeg => false,
        }
    }

    pub fn is_negative(&self) -> bool {
        return ! self.is_positive();
    }

    pub fn from_vector(v: &Vector4<Real>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for i in 0..Direction::POSITIVES.len() {
            let scalar_prod = v.dot(&Direction::POSITIVES[i].to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        match index_max {
            0 => {
                if is_positive {
                    return Direction::XPos;
                }
                else {
                    return Direction::XNeg;
                }
            },
            1 => {
                if is_positive {
                    return Direction::YPos;
                }
                else {
                    return Direction::YNeg;
                }
            },
            2 => {
                if is_positive {
                    return Direction::ZPos;
                }
                else {
                    return Direction::ZNeg;
                }
            },
            3 => {
                if is_positive {
                    return Direction::TPos;
                }
                else {
                    return Direction::TNeg;
                }
            },
            _ => {
                panic!("Implementiation error : invalide index")
            }
        }
    }

    pub fn to_positive(&self) -> Self {
        match self {
            Direction::XNeg => Direction::XPos,
            Direction::YNeg => Direction::YPos,
            Direction::ZNeg => Direction::ZPos,
            Direction::TNeg => Direction::TPos,
            _ => self.clone(),
        }
    }
    
    const POSITIVES: [Self; 4] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::TPos];
    const POSITIVES_SPACE: [Self; 3] = [Direction::XPos, Direction::YPos, Direction::ZPos];
}


impl From<Vector4<Real>> for Direction {
    fn from(v: Vector4<Real>) -> Self {
        Direction::from_vector(&v)
    }
}

impl From<Direction> for Vector4<Real> {
    fn from(d: Direction) -> Self {
        d.to_vector(1_f64)
    }
}
