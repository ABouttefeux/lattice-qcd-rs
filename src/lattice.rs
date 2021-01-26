
use na::{
    Vector4
};
use approx::*;
use super::Real;
use std::ops::{Index, IndexMut, Neg};

pub type PositiveF64 = Real;

/// a cyclique lattice in space. Does not store point and links but is used to generate them.
#[derive(Clone, Debug)]
pub struct LatticeCyclique {
    size: PositiveF64,
    dim: usize,
}

impl LatticeCyclique {
    /// we use the notation `[x, y, z, t]`
    pub const DIM: usize = 4;
    
    /// see [`LatticeLinkCanonical`], a conical link is a link whose direction is always positive.
    /// that means that a link form `[x, y, z, t]` with direction `-x`
    /// the link return is `[x - 1, y, z, t]` with direction `+x`
    pub fn get_link_canonical(&self, pos: &LatticePoint, dir: &Direction) -> LatticeLinkCanonical {
        let mut pos_link: LatticePoint = LatticePoint::new([0; LatticeCyclique::DIM]);
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
    
    pub fn get_link (&self, pos: &LatticePoint, dir: &Direction) -> LatticeLink {
        let mut pos_link = LatticePoint::new([0_usize; LatticeCyclique::DIM]);
        for i in 0..pos.len() {
            pos_link[i] = pos[i] % self.dim();
        }
        LatticeLink::new(pos_link, dir.clone())
    }
    
    /// transform a LatticeLink into a LatticeLinkCanonical, see
    /// [`LatticeCyclique::get_link_canonical`] and [`LatticeLinkCanonical`]
    pub fn get_canonical(&self, l: &LatticeLink) -> LatticeLinkCanonical{
        self.get_link_canonical(l.pos(), l.dir())
    }
    
    /// get the number of numbre of point in a single Direction.
    /// use [`LatticeCyclique::get_number_of_points`] for the total number of points.
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// get an Iterator over all canonical link oritend in space (i.e. no `t` direction)
    /// for a given time.
    pub fn get_links_space(&self, time_pos: usize) -> IteratorLatticeLinkCanonical {
        return IteratorLatticeLinkCanonical::new(&self, &self.get_link_canonical(&LatticePoint::from([0, 0, 0, time_pos]), Direction::POSITIVES_SPACE.first().unwrap()));
    }
    
    /// get an Iterator over all point for a given time.
    pub fn get_points(&self, time_pos: usize) -> IteratorLatticePoint {
        return IteratorLatticePoint::new(&self, &LatticePoint::from([0, 0, 0, time_pos]));
    }
    
    /// create a new lattice, size should be greater than 0 and dime greater or equal to 2
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
    
    /// total number of canonical links oriented in space for a set time
    pub fn get_number_of_canonical_links_space(&self) -> usize {
        self.get_number_of_points() * 3
    }
    
    /// total number of point in the lattice for a set time
    pub fn get_number_of_points(&self) -> usize {
        self.dim().pow(3)
    }
    
    pub fn size(&self) -> Real {
        self.size
    }
    
    pub fn add_point_direction(&self, mut l: LatticePoint, dir: &Direction) -> LatticePoint {
        if dir.is_positive() {
            l[dir.to_index()] = (l[dir.to_index()] + 1) % self.dim();
            return l;
        }
        else {
            let dir_pos = dir.to_positive();
            if l[dir_pos.to_index()] == 0 {
                l[dir_pos.to_index()] = self.dim() - 1;
            }
            else {
                l[dir_pos.to_index()] = (l[dir_pos.to_index()] - 1) % self.dim();
            }
            return l;
        }
    }
    
}

/// Iterator over [`LatticeLinkCanonical`]
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

/// Iterator over [`LatticePoint`]
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

/// we use the representation `[x, y, z, t]`
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LatticePoint {
    data: [usize; 4]
}

impl LatticePoint {
    pub fn new(data: [usize; 4]) -> Self {
        Self{data}
    }
    
    pub const fn len(&self) -> usize {
        self.data.len()
    }
}

impl Index<usize> for LatticePoint {
    type Output = usize;
    
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl IndexMut<usize> for LatticePoint {
        
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output{
        &mut self.data[pos]
    }
}

impl From<[usize; 4]> for LatticePoint {
    fn from(data: [usize; 4]) -> Self {
        LatticePoint::new(data)
    }
}

impl From<LatticePoint> for [usize; 4] {
    fn from(lattice_point: LatticePoint) -> Self {
        lattice_point.data
    }
}

impl From<LatticeLink> for LatticePoint {
    fn from(f: LatticeLink) -> Self{
        f.from
    }
}

/// Get a index for where the associated data is store in a vec
pub trait LatticeElementToIndex {
    fn to_index(&self, l: &LatticeCyclique) -> usize;
}

impl LatticeElementToIndex for LatticePoint {
    fn to_index(&self, l: &LatticeCyclique) -> usize {
        self[0] + self[1] * l.dim() + self[2] * l.dim().pow(2)
    }
}

impl LatticeElementToIndex for Direction {
    fn to_index(&self, _: &LatticeCyclique) -> usize {
        self.to_index()
    }
}

impl LatticeElementToIndex for LatticeLinkCanonical {
    fn to_index(&self, l: &LatticeCyclique) -> usize {
        self.pos().to_index(l) * 3 + self.dir().to_index()
    }
}

impl LatticeElementToIndex for usize {
    fn to_index(&self, _l: &LatticeCyclique) -> usize {
        self.clone()
    }
}

/// A canonical link of a lattice. It contain a position and a direction.
// The direction shoul always be positive.
/// This object can be used to safly index in a [`std::collections::HashMap`]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LatticeLinkCanonical {
    from: LatticePoint,
    dir: Direction,
}

impl LatticeLinkCanonical {
    pub fn new (from: LatticePoint, dir: Direction) -> Option<Self> {
        if dir.is_negative() {
            return None;
        }
        Some(Self {from, dir})
    }
    
    /// position of the link
    pub fn pos(&self) -> &LatticePoint{
        &self.from
    }
    
    pub fn pos_mut(&mut self) -> &mut LatticePoint{
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

/// A lattice link, contrary to [`LatticeLinkCanonical`] the direction can be negative.
/// This means that multiple link can be equivalent but does not have the same data
// and therefore hash (hopefully).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LatticeLink {
    from: LatticePoint,
    dir: Direction,
}

impl LatticeLink {
    pub fn new (from: LatticePoint, dir: Direction) -> Self {
        Self {from, dir}
    }
    
    pub fn pos(&self) -> &LatticePoint{
        &self.from
    }
    
    pub fn pos_mut(&mut self) -> &mut LatticePoint{
        &mut self.from
    }
    
    pub fn dir(&self) -> &Direction {
        &self.dir
    }
    
    pub fn dir_mut(&mut self) -> &mut Direction {
        &mut self.dir
    }
    
    /// get if the direction of the link is positive
    pub fn is_dir_positive(&self) -> bool {
        self.dir.is_positive()
    }
}

impl PartialEq<LatticeLink> for LatticeLinkCanonical {
    fn eq(&self, other: &LatticeLink) -> bool {
        *self.pos() == *other.pos() && *self.dir() == *other.dir()
    }
}

impl PartialEq<LatticeLinkCanonical> for LatticeLink{
    fn eq(&self, other: &LatticeLinkCanonical) -> bool {
        *other == *self
    }
}

/// Represent a sing
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

/// Represent a cardinal direction
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Direction {
    XPos, XNeg, YPos, YNeg, ZPos, ZNeg, TPos, TNeg,
}

impl Direction{
    
    pub const POSITIVES: [Self; 4] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::TPos];
    pub const POSITIVES_SPACE: [Self; 3] = [Direction::XPos, Direction::YPos, Direction::ZPos];
    pub const DIRECTIONS : [Self; 8] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::TPos, Direction::XNeg, Direction::YNeg, Direction::ZNeg, Direction::TNeg];
    pub const DIRECTIONS_SPACE : [Self; 6] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::XNeg, Direction::YNeg, Direction::ZNeg];
    
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
    /// find the direction the verctor point the most.
    /// for a zero vector return [`Direction::XPos`]
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
                // the code should attain this code. and therefore panicking is not expected.
                panic!("Implementiation error : invalide index")
            }
        }
    }
    
    /// return the positive direction associtated, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    pub fn to_positive(&self) -> Self {
        match self {
            Direction::XNeg => Direction::XPos,
            Direction::YNeg => Direction::YPos,
            Direction::ZNeg => Direction::ZPos,
            Direction::TNeg => Direction::TPos,
            _ => self.clone(),
        }
    }
    
    pub fn to_index(&self) -> usize {
        match self {
            Direction::XPos | Direction::XNeg => 0,
            Direction::YPos | Direction::YNeg => 1,
            Direction::ZPos | Direction::ZNeg => 2,
            Direction::TPos | Direction::TNeg => 3,
        }
    }
    
}

impl Neg for Direction{
    type Output = Self;
    
    fn neg(self) -> Self::Output{
        match self {
            Direction::XPos => Direction::XNeg,
            Direction::XNeg => Direction::XPos,
            Direction::YPos => Direction::YNeg,
            Direction::YNeg => Direction::YPos,
            Direction::ZPos => Direction::ZNeg,
            Direction::ZNeg => Direction::ZPos,
            Direction::TPos => Direction::TNeg,
            Direction::TNeg => Direction::TPos,
        }
    }
}

impl From<Direction> for usize {
    fn from(d: Direction) -> Self {
        d.to_index()
    }
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
