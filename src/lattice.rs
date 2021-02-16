
//! Defines lattices and lattice component.

use na::{
    Vector4,
    VectorN,
};
use approx::*;
use super::Real;
use std::ops::{Index, IndexMut, Neg};
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};


/// a cyclique lattice in space. Does not store point and links but is used to generate them.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeCyclique {
    size: Real,
    dim: usize,
}

impl LatticeCyclique {
    
    /// Number space + time dimension.
    ///
    /// Not to confuse with [`LatticeCyclique::dim`]. This is the dimension of space-time.
    pub const DIM_ST: usize = 4;
    
    /// see [`LatticeLinkCanonical`], a conical link is a link whose direction is always positive.
    /// That means that a link form `[x, y, z, t]` with direction `-x`
    /// the link return is `[x - 1, y, z, t]` (modulo the `lattice::dim()``) with direction `+x`
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclique, Direction, LatticePoint, LatticeLinkCanonical};
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// let point = LatticePoint::from([1, 0, 2, 0]);
    /// assert_eq!(
    ///     lattice.get_link_canonical(point, Direction::XNeg),
    ///     LatticeLinkCanonical::new(LatticePoint::from([0, 0, 2, 0]), Direction::XPos).unwrap()
    /// );
    /// assert_eq!(
    ///     lattice.get_link_canonical(point, Direction::XPos),
    ///     LatticeLinkCanonical::new(LatticePoint::from([1, 0, 2, 0]), Direction::XPos).unwrap()
    /// );
    /// assert_eq!(
    ///     lattice.get_link_canonical(point, Direction::YNeg),
    ///     LatticeLinkCanonical::new(LatticePoint::from([1, 3, 2, 0]), Direction::YPos).unwrap()
    /// );
    /// ```
    pub fn get_link_canonical(&self, pos: LatticePoint, dir: Direction) -> LatticeLinkCanonical {
        let mut pos_link: LatticePoint = pos;
        if ! dir.is_positive() {
            pos_link = self.add_point_direction(pos_link, &dir);
        }
        for i in 0..pos.len() {
            pos_link[i] %= self.dim();
        }
        LatticeLinkCanonical::new(pos_link, dir.to_positive()).unwrap()
    }
    
    /// Return a link build form `pos` and `dir`.
    ///
    /// It is similar to [`LatticeLink::new`]. It however enforce that the point is inside the bounds.
    /// If it is not, it will use the modulus of the bound.
    pub fn get_link (&self, pos: LatticePoint, dir: Direction) -> LatticeLink {
        let mut pos_link = LatticePoint::new([0_usize; LatticeCyclique::DIM_ST]);
        for i in 0..pos.len() {
            pos_link[i] = pos[i] % self.dim();
        }
        LatticeLink::new(pos_link, dir)
    }
    
    /// Transform a [`LatticeLink`] into a [`LatticeLinkCanonical`].
    ///
    /// See [`LatticeCyclique::get_link_canonical`] and [`LatticeLinkCanonical`].
    pub fn get_canonical(&self, l: LatticeLink) -> LatticeLinkCanonical {
        self.get_link_canonical(*l.pos(), *l.dir())
    }
    
    /// Get the number of points in a single direction.
    ///
    /// use [`LatticeCyclique::get_number_of_points`] for the total number of points.
    /// Not to confuse with [`LatticeCyclique::DIM_ST`]. This is the dimension of space-time.
    pub const fn dim(&self) -> usize {
        self.dim
    }
    
    /// Get an Iterator over all canonical link oriented in space (i.e. no `t` direction)
    /// for a given time.
    pub fn get_links_space(&self) -> IteratorLatticeLinkCanonical {
        return IteratorLatticeLinkCanonical::new(&self, self.get_link_canonical(LatticePoint::from([0; 4]), *Direction::POSITIVES_SPACE.first().unwrap()));
    }
    
    /// Get an Iterator over all point for a given time.
    pub fn get_points(&self) -> IteratorLatticePoint {
        IteratorLatticePoint::new(&self, LatticePoint::from([0; 4]))
    }
    
    /// create a new lattice with `size` the lattice size parameter, and `dim` the number of
    /// points in each spatial dimension.
    ///
    /// Size should be greater than 0 and dime greater or equal to 2.
    pub fn new(size: Real, dim: usize) -> Option<Self>{
        if size < 0_f64 {
            return None;
        }
        if dim < 2 {
            return None;
        }
        Some(Self {size, dim})
    }
    
    /// Total number of canonical links oriented in space for a set time.
    ///
    /// basically the number of element return by [`LatticeCyclique::get_links_space`]
    pub fn get_number_of_canonical_links_space(&self) -> usize {
        self.get_number_of_points() * 4
    }
    
    /// Total number of point in the lattice for a set time.
    pub fn get_number_of_points(&self) -> usize {
        self.dim().pow(4)
    }
    
    /// Return the lattice size factor.
    pub const fn size(&self) -> Real {
        self.size
    }
    
    /// get the next point in the lattice following the direction `dir`
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclique, Direction, LatticePoint};
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// let point = LatticePoint::from([1, 0, 2, 0]);
    /// assert_eq!(lattice.add_point_direction(point, &Direction::XPos), LatticePoint::from([2, 0, 2, 0]));
    /// // In the following case we get [_, 3, _, _] because `dim = 4`, and this lattice is cyclique.
    /// assert_eq!(lattice.add_point_direction(point, &Direction::YNeg), LatticePoint::from([1, 3, 2, 0]) );
    /// ```
    pub fn add_point_direction(&self, mut point: LatticePoint, dir: &Direction) -> LatticePoint {
        if dir.is_positive() {
            point[dir.to_index()] = (point[dir.to_index()] + 1) % self.dim();
        }
        else {
            let dir_pos = dir.to_positive();
            if point[dir_pos.to_index()] == 0 {
                point[dir_pos.to_index()] = self.dim() - 1;
            }
            else {
                point[dir_pos.to_index()] = (point[dir_pos.to_index()] - 1) % self.dim();
            }
        }
        point
    }
}

/// Iterator over [`LatticeLinkCanonical`] associated to a particular [`LatticeCyclique`].
/// Gives only spatial direction.
#[derive(Clone, Debug)]
pub struct IteratorLatticeLinkCanonical<'a> {
    lattice: &'a LatticeCyclique,
    element: Option<LatticeLinkCanonical>,
}

impl<'a> IteratorLatticeLinkCanonical<'a> {
    /// create a new iterator. The first [`IteratorLatticeLinkCanonical::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticeLinkCanonical, LatticeCyclique, LatticeLinkCanonical, LatticePoint, Direction};
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// let first_el = LatticeLinkCanonical::new(LatticePoint::from([1, 0, 2, 0]), Direction::YPos).unwrap();
    /// let mut iter = IteratorLatticeLinkCanonical::new(&lattice, first_el);
    /// assert_eq!(iter.next().unwrap(), first_el);
    /// ```
    pub const fn new(lattice: &'a LatticeCyclique, first_el: LatticeLinkCanonical) -> Self {
        Self {
            lattice,
            element: Some(first_el),
        }
    }
}

impl<'a> Iterator for IteratorLatticeLinkCanonical<'a> {
    type Item = LatticeLinkCanonical;
    
    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        const LINK_ARRAY: [Direction; 4] = Direction::POSITIVES;
        let previous_el = self.element;
        match &mut self.element {
            Some(element) => {
                let mut iter_dir = LINK_ARRAY.iter();
                iter_dir.find(|el| *el == element.dir());
                let new_dir = iter_dir.next();
                match new_dir {
                    Some(dir) => element.set_dir(*dir),
                    None => {
                        element.set_dir(*LINK_ARRAY.first().unwrap());
                        let mut iter = IteratorLatticePoint::new(self.lattice, *element.pos());
                        match iter.nth(1) { // get the second ellement
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
        previous_el
    }
}

/// Iterator over [`LatticePoint`]
#[derive(Clone, Debug)]
pub struct IteratorLatticePoint<'a>  {
    lattice: &'a LatticeCyclique,
    element: Option<LatticePoint>,
}

impl<'a> IteratorLatticePoint<'a> {
    /// create a new iterator. The first [`IteratorLatticePoint::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticePoint, LatticeCyclique, LatticePoint, Direction};
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// let first_el = LatticePoint::from([1, 0, 2, 0]);
    /// let mut iter = IteratorLatticePoint::new(&lattice, first_el);
    /// assert_eq!(iter.next().unwrap(), first_el);
    /// ```
    pub const fn new(lattice: &'a LatticeCyclique, first_el: LatticePoint) -> Self {
        Self {
            lattice,
            element: Some(first_el),
        }
    }
}


impl<'a> Iterator for IteratorLatticePoint<'a> {
    type Item = LatticePoint;
    
    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        let previous_el = self.element;
        match &mut self.element {
            Some(el) => {
                el[0] += 1;
                for i in 0..el.len() {
                    while el[i] >= self.lattice.dim() {
                        if i < el.len() - 1 { // every element exept the last one
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
        previous_el
    }
}

/// Represents point on a (any) lattice.
///
/// We use the representation `[x, y, z]`.
#[derive(Clone, Debug, Copy, PartialEq, Eq, PartialOrd, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticePoint {
    data: na::VectorN<usize, na::U4>
}

#[allow(clippy::len_without_is_empty)]
impl LatticePoint {
    
    /// Create a new point from the given coordinate.
    pub fn new(data: [usize; 4]) -> Self {
        Self {data : VectorN::from(data)}
    }
    
    /// Number of elements in [`LatticePoint`]. It is always 4
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// assert_eq!(LatticePoint::new([0; 4]).len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Index<usize> for LatticePoint {
    type Output = usize;
    
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound (greater or equal to 4)
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// let point = LatticePoint::new([0; 4]);
    /// let _ = point[4];
    /// ```
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl IndexMut<usize> for LatticePoint {
    
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound (greater or equal to 4)
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// let mut point = LatticePoint::new([0; 4]);
    /// point[4] += 1;
    /// ```
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
        lattice_point.data.into()
    }
}

impl From<&LatticePoint> for [usize; 4] {
    fn from(lattice_point: &LatticePoint) -> Self {
        lattice_point.data.into()
    }
}

/// Trait to convert an element on a lattice to an [`usize`].
///
/// Used mainly to index field on the lattice using [`std::vec::Vec`]
pub trait LatticeElementToIndex {
    /// Given a lattice return an index from the element
    fn to_index(&self, l: &LatticeCyclique) -> usize;
}

impl LatticeElementToIndex for LatticePoint {
    fn to_index(&self, l: &LatticeCyclique) -> usize {
        (self[0] % l.dim())
            + (self[1] % l.dim()) * l.dim()
            + (self[2] % l.dim()) * l.dim().pow(2)
            + (self[3] % l.dim()) * l.dim().pow(3)
    }
}

impl LatticeElementToIndex for Direction {
    /// equivalent to [`Direction::to_index()`]
    fn to_index(&self, _: &LatticeCyclique) -> usize {
        self.to_index()
    }
}

impl LatticeElementToIndex for LatticeLinkCanonical {
    fn to_index(&self, l: &LatticeCyclique) -> usize {
        self.pos().to_index(l) * 4 + self.dir().to_index()
    }
}

impl LatticeElementToIndex for usize {
    /// return self
    fn to_index(&self, _l: &LatticeCyclique) -> usize {
        *self
    }
}

/// A canonical link of a lattice. It contain a position and a direction.
///
/// The direction should always be positive.
/// By itself the link does not store data about the lattice. Hence most function require a [`LatticeCyclique`].
/// It also means that there is no guarantee that the object is inside a lattice.
/// You can use modulus over the elements to use inside a lattice.
///
/// This object can be used to safly index in a [`std::collections::HashMap`]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeLinkCanonical {
    from: LatticePoint,
    dir: Direction,
}

impl LatticeLinkCanonical {
    /// Try create a LatticeLinkCanonical. If the dir is negative it fails.
    ///
    /// To guaranty creating an element see [LatticeCyclique::get_link_canonical].
    /// The creation of an element this ways does not guaranties that the element is inside a lattice.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, Direction};
    /// let l = LatticeLinkCanonical::new(LatticePoint::new([0; 4]), Direction::XNeg);
    /// assert_eq!(l, None);
    ///
    /// let l = LatticeLinkCanonical::new(LatticePoint::new([0; 4]), Direction::XPos);
    /// assert!(l.is_some());
    /// ```
    pub const fn new (from: LatticePoint, dir: Direction) -> Option<Self> {
        if dir.is_negative() {
            return None;
        }
        Some(Self {from, dir})
    }
    
    /// position of the link
    pub const fn pos(&self) -> &LatticePoint{
        &self.from
    }
    
    pub fn pos_mut(&mut self) -> &mut LatticePoint{
        &mut self.from
    }
    
    /// Direction of the link.
    pub const fn dir(&self) -> &Direction {
        &self.dir
    }
    
    /// Set the direction to dir
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, Direction};
    /// let mut lattice_link_canonical = LatticeLinkCanonical::new(LatticePoint::new([0; 4]), Direction::YPos).unwrap();
    /// lattice_link_canonical.set_dir(Direction::XPos);
    /// assert_eq!(*lattice_link_canonical.dir(), Direction::XPos);
    /// ```
    /// # Panic
    /// panic if a negative direction is given.
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, Direction};
    /// # let mut lattice_link_canonical = LatticeLinkCanonical::new(LatticePoint::new([0; 4]), Direction::XPos).unwrap();
    /// lattice_link_canonical.set_dir(Direction::XNeg);
    /// ```
    pub fn set_dir(&mut self, dir: Direction) {
        if dir.is_negative(){
            panic!("Cannot set a negative direction to a canonical link.");
        }
        self.dir = dir;
    }
    
    /// Set the direction using positive direction. i.e. if a direction `-x` is passed
    /// the direction assigned will be `+x`.
    ///
    /// This is equivalent to `link.set_dir(dir.to_positive())`.
    pub fn set_dir_positive(&mut self, dir: Direction) {
        self.dir = dir.to_positive();
    }
}

impl From<LatticeLinkCanonical> for LatticeLink {
    fn from(l : LatticeLinkCanonical) -> Self {
        LatticeLink::new(l.from, l.dir)
    }
}

impl From<&LatticeLinkCanonical> for LatticeLink {
    fn from(l : &LatticeLinkCanonical) -> Self {
        LatticeLink::new(l.from, l.dir)
    }
}

/// A lattice link, contrary to [`LatticeLinkCanonical`] the direction can be negative.
///
/// This means that multiple link can be equivalent but does not have the same data
/// and therefore hash (hopefully).
///
/// By itself the link does not store data about the lattice. Hence most function require a [`LatticeCyclique`].
/// It also means that there is no guarantee that the object is inside a lattice.
/// You can use modulus over the elements to use inside a lattice.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeLink {
    from: LatticePoint,
    dir: Direction,
}

impl LatticeLink {
    /// Create a link from position `from` and direction `dir`.
    pub const fn new (from: LatticePoint, dir: Direction) -> Self {
        Self {from, dir}
    }
    
    /// Get the position of the link.
    pub const fn pos(&self) -> &LatticePoint{
        &self.from
    }
    
    /// Get a mutable reference to the position of the link.
    pub fn pos_mut(&mut self) -> &mut LatticePoint{
        &mut self.from
    }
    
    /// Get the direction of the link.
    pub const fn dir(&self) -> &Direction {
        &self.dir
    }
    
    /// Get a mutable reference to the direction of the link.
    pub fn dir_mut(&mut self) -> &mut Direction {
        &mut self.dir
    }
    
    /// Get if the direction of the link is positive.
    pub const fn is_dir_positive(&self) -> bool {
        self.dir.is_positive()
    }
    
    /// Get if the direction of the link is negative.
    pub const fn is_dir_negative(&self) -> bool {
        self.dir.is_negative()
    }
}

/* removed for being potentially confusing
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
*/

/// Represent a sing
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum Sign {
    Negative, Positive, Zero
}

impl Sign {
    /// return a f64 form the sign `(-1_f64, 0_f64, 1_f64)`.
    pub const fn to_f64(self) -> f64 {
        match self {
            Sign::Negative => -1_f64,
            Sign::Positive => 1_f64,
            Sign::Zero => 0_f64,
        }
    }
    
    /// Get the sign form a f64.
    ///
    /// If the value is very close to zero but not quite the sing will nonetheless be Sign::Zero.
    pub fn sign(f: f64) -> Self {
        if relative_eq!(f, 0_f64) {
            return Sign::Zero;
        }
        else if f > 0_f64{
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum Direction {
    XPos, XNeg, YPos, YNeg, ZPos, ZNeg, TPos, TNeg,
}

impl Direction{
    
    /// List of all positives directions.
    pub const POSITIVES: [Self; 4] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::TPos];
    
    /// List spatial positive direction.
    pub const POSITIVES_SPACE: [Self; 3] = [Direction::XPos, Direction::YPos, Direction::ZPos];
    
    /// List all directions.
    pub const DIRECTIONS : [Self; 8] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::TPos, Direction::XNeg, Direction::YNeg, Direction::ZNeg, Direction::TNeg];
    
    /// List all spatial directions.
    pub const DIRECTIONS_SPACE : [Self; 6] = [Direction::XPos, Direction::YPos, Direction::ZPos, Direction::XNeg, Direction::YNeg, Direction::ZNeg];
    
    /// Convert the direction into a vector of norm `a`;
    pub fn to_vector(&self, a: f64) -> Vector4<Real> {
        self.to_unit_vector() * a
    }
    
    /// Convert the direction into a vector of norm `1`;
    pub fn to_unit_vector(&self) -> Vector4<Real> {
        match self {
            Direction::XPos => Vector4::<Real>::new(1_f64, 0_f64, 0_f64, 0_f64),
            Direction::XNeg => Vector4::<Real>::new(-1_f64, 0_f64, 0_f64, 0_f64),
            Direction::YPos => Vector4::<Real>::new(0_f64, 1_f64, 0_f64, 0_f64),
            Direction::YNeg => Vector4::<Real>::new(0_f64, -1_f64, 0_f64, 0_f64),
            Direction::ZPos => Vector4::<Real>::new(0_f64, 0_f64, 1_f64, 0_f64),
            Direction::ZNeg => Vector4::<Real>::new(0_f64, 0_f64, -1_f64, 0_f64),
            Direction::TPos => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, 1_f64),
            Direction::TNeg => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, -1_f64),
        }
    }
    
    /// Get if the position is positive.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// assert_eq!( Direction::XPos.is_positive(), true);
    /// assert_eq!( Direction::TPos.is_positive(), true);
    /// assert_eq!( Direction::YNeg.is_positive(), false);
    /// ```
    pub const fn is_positive(&self) -> bool {
        match self {
            Direction::XPos | Direction::YPos | Direction::ZPos | Direction::TPos => true,
            Direction::XNeg | Direction::YNeg | Direction::ZNeg | Direction::TNeg => false,
        }
    }
    
    /// Get if the position is Negative. see [`Direction::is_positive`]
    pub const fn is_negative(&self) -> bool {
        ! self.is_positive()
    }
    
    /// Find the direction the vector point the most.
    /// For a zero vector return [`Direction::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// # extern crate nalgebra;
    /// assert_eq!(Direction::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)), Direction::XPos);
    /// assert_eq!(Direction::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)), Direction::YNeg);
    /// assert_eq!(Direction::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)), Direction::TPos);
    /// ```
    pub fn from_vector(v: &Vector4<Real>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        // TODO try fold ?
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
                unreachable!("Implementiation error : invalide index")
            }
        }
    }
    
    /// Return the positive direction associated, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// assert_eq!(Direction::XNeg.to_positive(), Direction::XPos);
    /// assert_eq!(Direction::YPos.to_positive(), Direction::YPos);
    /// ```
    pub const fn to_positive(self) -> Self {
        match self {
            Direction::XNeg => Direction::XPos,
            Direction::YNeg => Direction::YPos,
            Direction::ZNeg => Direction::ZPos,
            Direction::TNeg => Direction::TPos,
            _ => self,
        }
    }
    
    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// assert_eq!(Direction::XPos.to_index(), 0);
    /// assert_eq!(Direction::XNeg.to_index(), 0);
    /// assert_eq!(Direction::YPos.to_index(), 1);
    /// assert_eq!(Direction::YNeg.to_index(), 1);
    /// assert_eq!(Direction::ZPos.to_index(), 2);
    /// assert_eq!(Direction::ZNeg.to_index(), 2);
    /// assert_eq!(Direction::TPos.to_index(), 3);
    /// assert_eq!(Direction::TNeg.to_index(), 3);
    /// ```
    pub const fn to_index(&self) -> usize {
        match self {
            Direction::XPos | Direction::XNeg => 0,
            Direction::YPos | Direction::YNeg => 1,
            Direction::ZPos | Direction::ZNeg => 2,
            Direction::TPos | Direction::TNeg => 3,
        }
    }
}

/// Return the negative of a direction
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::Direction;
/// assert_eq!(- Direction::XNeg, Direction::XPos);
/// assert_eq!(- Direction::YPos, Direction::YNeg);
/// ```
impl Neg for Direction {
    type Output = Self;
    
    fn neg(self) -> Self::Output {
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

impl Neg for &Direction {
    type Output = Self;
    
    fn neg(self) -> Self::Output {
        match self {
            Direction::XPos => &Direction::XNeg,
            Direction::XNeg => &Direction::XPos,
            Direction::YPos => &Direction::YNeg,
            Direction::YNeg => &Direction::YPos,
            Direction::ZPos => &Direction::ZNeg,
            Direction::ZNeg => &Direction::ZPos,
            Direction::TPos => &Direction::TNeg,
            Direction::TNeg => &Direction::TPos,
        }
    }
}

/// Return [`Direction::to_index`].
impl From<Direction> for usize {
    fn from(d: Direction) -> Self {
        d.to_index()
    }
}

/// Return [`Direction::to_index`].
impl From<&Direction> for usize {
    fn from(d: &Direction) -> Self {
        d.to_index()
    }
}

/// Return [`Direction::from_vector`].
impl From<Vector4<Real>> for Direction {
    fn from(v: Vector4<Real>) -> Self {
        Direction::from_vector(&v)
    }
}

/// Return [`Direction::from_vector`].
impl From<&Vector4<Real>> for Direction {
    fn from(v: &Vector4<Real>) -> Self {
        Direction::from_vector(v)
    }
}

/// Return [`Direction::to_unit_vector`].
impl From<Direction> for Vector4<Real> {
    fn from(d: Direction) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`Direction::to_unit_vector`].
impl From<&Direction> for Vector4<Real> {
    fn from(d: &Direction) -> Self {
        d.to_unit_vector()
    }
}
