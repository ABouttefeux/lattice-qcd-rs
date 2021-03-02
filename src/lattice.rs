
//! Defines lattices and lattice component.

use na::{
    Vector4,
    VectorN,
    base::dimension::{DimName},
    base::allocator::Allocator,
    DefaultAllocator,
};
use approx::*;
use super::{
    Real,
    dim::*,
};
use std::ops::{Index, IndexMut, Neg};
#[cfg(feature = "serde-serialize")]
use serde::{Serialize, Deserialize};
use std::marker::PhantomData;
use std::convert::{TryInto};
use lattice_qcd_rs_procedural_macro::implement_direction_list;

/// a cyclique lattice in space. Does not store point and links but is used to generate them.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeCyclique<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
{
    size: Real,
    dim: usize,
    #[cfg_attr(feature = "serde-serialize", serde(skip) )]
    _phantom: PhantomData<D>,
}

impl<D> LatticeCyclique<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    
    /// Number space + time dimension.
    ///
    /// Not to confuse with [`LatticeCyclique::dim`]. This is the dimension of space-time.
    pub fn dim_st() -> usize {
        D::dim()
    }
    
    /// see [`LatticeLinkCanonical`], a conical link is a link whose direction is always positive.
    /// That means that a link form `[x, y, z, t]` with direction `-x`
    /// the link return is `[x - 1, y, z, t]` (modulo the `lattice::dim()``) with direction `+x`
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclique, DirectionEnum, LatticePoint, LatticeLinkCanonical};
    /// # use lattice_qcd_rs::dim::U4;
    /// let lattice = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
    /// let point = LatticePoint::<U4>::new([1, 0, 2, 0].into());
    /// assert_eq!(
    ///     lattice.get_link_canonical(point, DirectionEnum::XNeg.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([0, 0, 2, 0].into()), DirectionEnum::XPos.into()).unwrap()
    /// );
    /// assert_eq!(
    ///     lattice.get_link_canonical(point, DirectionEnum::XPos.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([1, 0, 2, 0].into()), DirectionEnum::XPos.into()).unwrap()
    /// );
    /// assert_eq!(
    ///     lattice.get_link_canonical(point, DirectionEnum::YNeg.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([1, 3, 2, 0].into()), DirectionEnum::YPos.into()).unwrap()
    /// );
    /// ```
    pub fn get_link_canonical(&self, pos: LatticePoint<D>, dir: Direction<D>) -> LatticeLinkCanonical<D> {
        let mut pos_link = pos;
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
    pub fn get_link (&self, pos: LatticePoint<D>, dir: Direction<D>) -> LatticeLink<D> {
        let mut pos_link = LatticePoint::new_zero();
        for i in 0..pos.len() {
            pos_link[i] = pos[i] % self.dim();
        }
        LatticeLink::new(pos_link, dir)
    }
    
    /// Transform a [`LatticeLink`] into a [`LatticeLinkCanonical`].
    ///
    /// See [`LatticeCyclique::get_link_canonical`] and [`LatticeLinkCanonical`].
    pub fn get_canonical(&self, l: LatticeLink<D>) -> LatticeLinkCanonical<D> {
        self.get_link_canonical(*l.pos(), *l.dir())
    }
    
    /// Get the number of points in a single direction.
    ///
    /// use [`LatticeCyclique::get_number_of_points`] for the total number of points.
    /// Not to confuse with [`LatticeCyclique::dim_st`]. This is the dimension of space-time.
    pub fn dim(&self) -> usize {
        self.dim
    }
    
    /// Get an Iterator over all canonical link
    pub fn get_links(&self) -> IteratorLatticeLinkCanonical<'_, D> {
        return IteratorLatticeLinkCanonical::new(&self, self.get_link_canonical(LatticePoint::new_zero(), *Direction::get_all_positive_directions().first().unwrap()));
    }
    
    /// Get an Iterator over all points.
    pub fn get_points(&self) -> IteratorLatticePoint<'_, D> {
        IteratorLatticePoint::new(&self, LatticePoint::new_zero())
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
        Some(Self {size, dim, _phantom: PhantomData})
    }
    
    /// Total number of canonical links oriented in space for a set time.
    ///
    /// basically the number of element return by [`LatticeCyclique::get_links`]
    pub fn get_number_of_canonical_links_space(&self) -> usize {
        self.get_number_of_points() * D::dim()
    }
    
    /// Total number of point in the lattice for a set time.
    pub fn get_number_of_points(&self) -> usize {
        self.dim().pow(D::dim().try_into().unwrap())
    }
    
    /// Return the lattice size factor.
    pub fn size(&self) -> Real {
        self.size
    }
    
    /// get the next point in the lattice following the direction `dir`
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclique, DirectionEnum, LatticePoint};
    /// # use lattice_qcd_rs::dim::U4;
    /// let lattice = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
    /// let point = LatticePoint::<U4>::from([1, 0, 2, 0]);
    /// assert_eq!(lattice.add_point_direction(point, &DirectionEnum::XPos.into()), LatticePoint::from([2, 0, 2, 0]));
    /// // In the following case we get [_, 3, _, _] because `dim = 4`, and this lattice is cyclique.
    /// assert_eq!(lattice.add_point_direction(point, &DirectionEnum::YNeg.into()), LatticePoint::from([1, 3, 2, 0]));
    /// ```
    pub fn add_point_direction(&self, mut point: LatticePoint<D>, dir: &Direction<D>) -> LatticePoint<D> {
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
pub struct IteratorLatticeLinkCanonical<'a, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
{
    lattice: &'a LatticeCyclique<D>,
    element: Option<LatticeLinkCanonical<D>>,
}

impl<'a, D> IteratorLatticeLinkCanonical<'a, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// create a new iterator. The first [`IteratorLatticeLinkCanonical::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticeLinkCanonical, LatticeCyclique, LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// # use lattice_qcd_rs::dim::U4;
    /// let lattice = LatticeCyclique::<U4>::new(1_f64, 4).unwrap();
    /// let first_el = LatticeLinkCanonical::<U4>::new(LatticePoint::from([1, 0, 2, 0]), DirectionEnum::YPos.into()).unwrap();
    /// let mut iter = IteratorLatticeLinkCanonical::new(&lattice, first_el);
    /// assert_eq!(iter.next().unwrap(), first_el);
    /// ```
    pub fn new(lattice: &'a LatticeCyclique<D>, first_el: LatticeLinkCanonical<D>) -> Self {
        Self {
            lattice,
            element: Some(first_el),
        }
    }
}

impl<'a, D> Iterator for IteratorLatticeLinkCanonical<'a, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Item = LatticeLinkCanonical<D>;
    
    // TODO improve
    fn next(&mut self) -> Option<Self::Item> {
        let link_iter = Direction::get_all_positive_directions();
        let previous_el = self.element;
        match &mut self.element {
            Some(element) => {
                let mut iter_dir = link_iter.iter();
                iter_dir.find(|el| *el == element.dir());
                let new_dir = iter_dir.next();
                match new_dir {
                    Some(dir) => element.set_dir(*dir),
                    None => {
                        element.set_dir(*link_iter.first().unwrap());
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
pub struct IteratorLatticePoint<'a, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
{
    lattice: &'a LatticeCyclique<D>,
    element: Option<LatticePoint<D>>,
}

impl<'a, D> IteratorLatticePoint<'a, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// create a new iterator. The first [`IteratorLatticePoint::next()`] will return `first_el`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{IteratorLatticePoint, LatticeCyclique, LatticePoint, Direction};
    /// let lattice = LatticeCyclique::new(1_f64, 4).unwrap();
    /// let first_el = LatticePoint::from([1, 0, 2, 0]);
    /// let mut iter = IteratorLatticePoint::new(&lattice, first_el);
    /// assert_eq!(iter.next().unwrap(), first_el);
    /// ```
    pub fn new(lattice: &'a LatticeCyclique<D>, first_el: LatticePoint<D>) -> Self {
        Self {
            lattice,
            element: Some(first_el),
        }
    }
}


impl<'a, D> Iterator for IteratorLatticePoint<'a, D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    type Item = LatticePoint<D>;
    
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
pub struct LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D> : Copy,
{
    #[cfg_attr(feature = "serde-serialize", serde(bound(serialize = "VectorN<usize, D>: Serialize", deserialize = "VectorN<usize, D>: Deserialize<'de>")) )]
    data: na::VectorN<usize, D>
}

impl<'a, D> IntoIterator for &'a LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D> : Copy,
{
    type Item = &'a usize;
    type IntoIter = <&'a VectorN<usize, D> as IntoIterator>::IntoIter;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<D> LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    
    pub fn new(data: VectorN<usize, D>) -> Self {
        Self {data}
    }
    
    pub fn new_zero() -> Self{
        Self {data: VectorN::zeros()}
    }
    
    pub fn from_fn<F>(mut f: F) -> Self
    where F: FnMut(usize) -> usize {
        Self::new(VectorN::from_fn(|index, _| f(index) ))
    }
    
    /// Number of elements in [`LatticePoint`].
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    #[allow(clippy::unused_self)]
    pub fn is_empty(&self) -> bool {
        D::dim() == 0
    }
    
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.data.iter()
    }
}

impl<D> Index<usize> for LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    type Output = usize;
    
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// let point = LatticePoint::new([0; 4].into());
    /// let _ = point[4];
    /// ```
    fn index(&self, pos: usize) -> &Self::Output{
        &self.data[pos]
    }
}

impl<D> IndexMut<usize> for LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    
    /// Get the element at position `pos`
    /// # Panic
    /// Panics if the position is out of bound
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::LatticePoint;
    /// let mut point = LatticePoint::new([0; 4].into());
    /// point[4] += 1;
    /// ```
    fn index_mut(&mut self, pos: usize) -> &mut Self::Output{
        &mut self.data[pos]
    }
}

impl<D, T> From<T> for LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + From<T>,
{
    fn from(data: T) -> Self {
        LatticePoint::new(VectorN::from(data))
    }
}


macro_rules! implement_from_lattice_point{
    ($(($l:literal, $i:ident)) ,+) => {
        $(
            impl From<LatticePoint<$i>> for [usize; $l] {
                fn  from(data: LatticePoint<$i>) -> [usize; $l] {
                    data.data.into()
                }
            }
        )*
    }
}

implement_from_lattice_point!((1, U1), (2, U2), (3, U3), (4, U4), (5, U5), (6, U6), (7, U7), (8, U8), (9, U9), (10, U10));

/// Trait to convert an element on a lattice to an [`usize`].
///
/// Used mainly to index field on the lattice using [`std::vec::Vec`]
pub trait LatticeElementToIndex<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// Given a lattice return an index from the element
    fn to_index(&self, l: &LatticeCyclique<D>) -> usize;
}

impl<D> LatticeElementToIndex<D> for LatticePoint<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn to_index(&self, l: &LatticeCyclique<D>) -> usize {
        self.iter().enumerate().map(|(index, pos)| {
            (pos % l.dim()) * l.dim().pow(index.try_into().unwrap())
        }).sum()
    }
}

impl<D> LatticeElementToIndex<D> for Direction<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// equivalent to [`Direction::to_index()`]
    fn to_index(&self, _: &LatticeCyclique<D>) -> usize {
        self.to_index()
    }
}

impl<D> LatticeElementToIndex<D> for LatticeLinkCanonical<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    fn to_index(&self, l: &LatticeCyclique<D>) -> usize {
        self.pos().to_index(l) * D::dim() + self.dir().to_index()
    }
}

impl<D> LatticeElementToIndex<D> for usize
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy + Send + Sync,
    Direction<D>: DirectionList,
{
    /// return self
    fn to_index(&self, _l: &LatticeCyclique<D>) -> usize {
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
pub struct LatticeLinkCanonical<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    #[cfg_attr(feature = "serde-serialize", serde(bound(serialize = "VectorN<usize, D>: Serialize", deserialize = "VectorN<usize, D>: Deserialize<'de>")) )]
    from: LatticePoint<D>,
    dir: Direction<D>,
}


impl<D> LatticeLinkCanonical<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    /// Try create a LatticeLinkCanonical. If the dir is negative it fails.
    ///
    /// To guaranty creating an element see [LatticeCyclique::get_link_canonical].
    /// The creation of an element this ways does not guaranties that the element is inside a lattice.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// let l = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XNeg.into());
    /// assert_eq!(l, None);
    ///
    /// let l = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XPos.into());
    /// assert!(l.is_some());
    /// ```
    pub fn new (from: LatticePoint<D>, dir: Direction<D>) -> Option<Self> {
        if dir.is_negative() {
            return None;
        }
        Some(Self {from, dir})
    }
    
    /// position of the link
    pub fn pos(&self) -> &LatticePoint<D>{
        &self.from
    }
    
    pub fn pos_mut(&mut self) -> &mut LatticePoint<D>{
        &mut self.from
    }
    
    /// Direction of the link.
    pub fn dir(&self) -> &Direction<D> {
        &self.dir
    }
    
    /// Set the direction to dir
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// let mut lattice_link_canonical = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::YPos.into()).unwrap();
    /// lattice_link_canonical.set_dir(DirectionEnum::XPos.into());
    /// assert_eq!(*lattice_link_canonical.dir(), DirectionEnum::XPos.into());
    /// ```
    /// # Panic
    /// panic if a negative direction is given.
    /// ```should_panic
    /// # use lattice_qcd_rs::lattice::{LatticeLinkCanonical, LatticePoint, DirectionEnum};
    /// # let mut lattice_link_canonical = LatticeLinkCanonical::new(LatticePoint::new([0; 4].into()), DirectionEnum::XPos.into()).unwrap();
    /// lattice_link_canonical.set_dir(DirectionEnum::XNeg.into());
    /// ```
    pub fn set_dir(&mut self, dir: Direction<D>) {
        if dir.is_negative(){
            panic!("Cannot set a negative direction to a canonical link.");
        }
        self.dir = dir;
    }
    
    /// Set the direction using positive direction. i.e. if a direction `-x` is passed
    /// the direction assigned will be `+x`.
    ///
    /// This is equivalent to `link.set_dir(dir.to_positive())`.
    pub fn set_dir_positive(&mut self, dir: Direction<D>) {
        self.dir = dir.to_positive();
    }
}

impl<D> From<LatticeLinkCanonical<D>> for LatticeLink<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    fn from(l : LatticeLinkCanonical<D>) -> Self {
        LatticeLink::new(l.from, l.dir)
    }
}

impl<D> From<&LatticeLinkCanonical<D>> for LatticeLink<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    fn from(l : &LatticeLinkCanonical<D>) -> Self {
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
pub struct LatticeLink<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    #[cfg_attr(feature = "serde-serialize", serde(bound(serialize = "VectorN<usize, D>: Serialize", deserialize = "VectorN<usize, D>: Deserialize<'de>")) )]
    from: LatticePoint<D>,
    dir: Direction<D>,
}

impl<D> LatticeLink<D>
    where D: DimName,
    DefaultAllocator: Allocator<usize, D>,
    VectorN<usize, D>: Copy,
{
    /// Create a link from position `from` and direction `dir`.
    pub fn new (from: LatticePoint<D>, dir: Direction<D>) -> Self {
        Self {from, dir}
    }
    
    /// Get the position of the link.
    pub fn pos(&self) -> &LatticePoint<D>{
        &self.from
    }
    
    /// Get a mutable reference to the position of the link.
    pub fn pos_mut(&mut self) -> &mut LatticePoint<D>{
        &mut self.from
    }
    
    /// Get the direction of the link.
    pub fn dir(&self) -> &Direction<D> {
        &self.dir
    }
    
    /// Get a mutable reference to the direction of the link.
    pub fn dir_mut(&mut self) -> &mut Direction<D> {
        &mut self.dir
    }
    
    /// Get if the direction of the link is positive.
    pub fn is_dir_positive(&self) -> bool {
        self.dir.is_positive()
    }
    
    /// Get if the direction of the link is negative.
    pub fn is_dir_negative(&self) -> bool {
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

impl Neg for Sign {
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Zero => Sign::Zero,
            Sign::Negative => Sign::Positive,
        }
    }
}

/// Represent a cardinal direction
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Direction<D>
    where D: DimName
{
    index_dir: usize,
    is_positive: bool,
    #[cfg_attr(feature = "serde-serialize", serde(skip) )]
    _phantom: PhantomData<D>,
}

impl<D> Direction<D>
    where D: DimName
{
    pub fn new(index_dir: usize, is_positive: bool) -> Option<Self> {
        if index_dir >= D::dim() {
            return None;
        }
        Some(Self {index_dir, is_positive, _phantom: PhantomData})
    }
    
    /// List of all positives directions.
    /// This is very slow use [`DirectionList::get_all_positive_directions`] instead
    pub fn positives_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(D::dim());
        for i in 0..D::dim() {
            x.push(Self::new(i, true).unwrap());
        }
        x
    }
    
    /// List all directions.
    /// This is very slow use [`DirectionList::get_all_directions`] instead
    pub fn directions_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(2 * D::dim());
        for i in 0..D::dim() {
            x.push(Self::new(i, true).unwrap());
            x.push(Self::new(i, false).unwrap());
        }
        x
    }
    
    
    /// Get if the position is positive.
    pub fn is_positive(&self) -> bool {
        self.is_positive
    }
    
    /// Get if the position is Negative. see [`Direction::is_positive`]
    pub fn is_negative(&self) -> bool {
        ! self.is_positive()
    }
    
    /// Return the positive direction associated, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// # use lattice_qcd_rs::dim::U4;
    /// assert_eq!(Direction::<U4>::new(1, false).unwrap().to_positive(), Direction::<U4>::new(1, true).unwrap());
    /// assert_eq!(Direction::<U4>::new(1, true).unwrap().to_positive(), Direction::<U4>::new(1, true).unwrap());
    /// ```
    pub fn to_positive(mut self) -> Self {
        self.is_positive = true;
        self
    }
    
    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// # use lattice_qcd_rs::dim::U4;
    /// assert_eq!(Direction::<U4>::new(1, false).unwrap().to_index(), 1);
    /// ```
    pub fn to_index(&self) -> usize {
        self.index_dir
    }
}

impl<D> Direction<D>
    where D: DimName,
    DefaultAllocator: Allocator<Real, D>,
{
    
    /// Convert the direction into a vector of norm `a`;
    pub fn to_vector(&self, a: f64) -> VectorN<Real, D> {
        self.to_unit_vector() * a
    }
    
    pub fn dim() -> usize {
        D::dim()
    }
    
    /// Convert the direction into a vector of norm `1`;
    pub fn to_unit_vector(&self) -> VectorN<Real, D> {
        let mut v = VectorN::zeros();
        v[self.index_dir] = 1_f64;
        v
    }
    
}


impl<D> Direction<D>
    where D: DimName,
    DefaultAllocator: Allocator<Real, D>,
    Direction<D>: DirectionList,
{
    
    
    
    /// Find the direction the vector point the most.
    /// For a zero vector return [`DirectionEnum::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// # use lattice_qcd_rs::dim::U4;
    /// # extern crate nalgebra;
    /// assert_eq!(Direction::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)), Direction::<U4>::new(0, true).unwrap());
    /// assert_eq!(Direction::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)), Direction::<U4>::new(1, false).unwrap());
    /// assert_eq!(Direction::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)), Direction::<U4>::new(3, true).unwrap());
    /// ```
    pub fn from_vector(v: &VectorN<Real, D>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for (i, dir) in Self::get_all_positive_directions().iter().enumerate() {
            let scalar_prod = v.dot(&dir.to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        Self::new(index_max, is_positive).expect("Unreachable")
    }
}
/// List all possible direction
pub trait DirectionList: Sized {
    /// List all directions.
    fn get_all_directions()->& 'static [Self];
    /// List all positive directions.
    fn get_all_positive_directions()->& 'static [Self];
}

implement_direction_list!();

impl<D: DimName> Neg for Direction<D> {
    type Output = Self;
    
    fn neg(mut self) -> Self::Output {
        self.is_positive = ! self.is_positive;
        self
    }
}

impl<D: DimName> Neg for &Direction<D> {
    type Output = Direction<D>;
    
    fn neg(self) -> Self::Output {
        - *self
    }
}

/// Return [`Direction::to_index`].
impl<D: DimName> From<Direction<D>> for usize {
    fn from(d: Direction<D>) -> Self {
        d.to_index()
    }
}

/// Return [`Direction::to_index`].
impl<D: DimName> From<&Direction<D>> for usize {
    fn from(d: &Direction<D>) -> Self {
        d.to_index()
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<D> From<VectorN<Real, D>> for Direction<D>
    where D: DimName,
    DefaultAllocator: Allocator<Real, D>,
    Direction<D>: DirectionList,
{
    fn from(v: VectorN<Real, D>) -> Self {
        Direction::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<D> From<&VectorN<Real, D>> for Direction<D>
    where D: DimName,
    DefaultAllocator: Allocator<Real, D>,
    Direction<D>: DirectionList,
{
    fn from(v: &VectorN<Real, D>) -> Self {
        Direction::<D>::from_vector(v)
    }
}

/// Return [`Direction::to_unit_vector`].
impl<D> From<Direction<D>> for VectorN<Real, D>
    where D: DimName,
    DefaultAllocator: Allocator<Real, D>,
{
    fn from(d: Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`Direction::to_unit_vector`].
impl<D> From<&Direction<D>> for VectorN<Real, D>
    where D: DimName,
    DefaultAllocator: Allocator<Real, D>,
{
    fn from(d: &Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

impl From<DirectionEnum> for Direction<na::U4> {
    fn from(d: DirectionEnum) -> Self {
        Self::new(d.to_index(), d.is_positive()).expect("unreachable")
    }
}

impl From<&DirectionEnum> for Direction<na::U4> {
    fn from(d: &DirectionEnum) -> Self {
        Self::new(d.to_index(), d.is_positive()).expect("unreachable")
    }
}

/// Represent a cardinal direction
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum DirectionEnum {
    XPos, XNeg, YPos, YNeg, ZPos, ZNeg, TPos, TNeg,
}

impl DirectionEnum {
    
    /// List of all positives directions.
    pub const POSITIVES: [Self; 4] = [DirectionEnum::XPos, DirectionEnum::YPos, DirectionEnum::ZPos, DirectionEnum::TPos];
    
    /// List spatial positive direction.
    pub const POSITIVES_SPACE: [Self; 3] = [DirectionEnum::XPos, DirectionEnum::YPos, DirectionEnum::ZPos];
    
    /// List all directions.
    pub const DIRECTIONS : [Self; 8] = [DirectionEnum::XPos, DirectionEnum::YPos, DirectionEnum::ZPos, DirectionEnum::TPos, DirectionEnum::XNeg, DirectionEnum::YNeg, DirectionEnum::ZNeg, DirectionEnum::TNeg];
    
    /// List all spatial directions.
    pub const DIRECTIONS_SPACE : [Self; 6] = [DirectionEnum::XPos, DirectionEnum::YPos, DirectionEnum::ZPos, DirectionEnum::XNeg, DirectionEnum::YNeg, DirectionEnum::ZNeg];
    
    /// Convert the direction into a vector of norm `a`;
    pub fn to_vector(self, a: f64) -> Vector4<Real> {
        self.to_unit_vector() * a
    }
    
    /// Convert the direction into a vector of norm `1`;
    pub fn to_unit_vector(self) -> Vector4<Real> {
        match self {
            DirectionEnum::XPos => Vector4::<Real>::new(1_f64, 0_f64, 0_f64, 0_f64),
            DirectionEnum::XNeg => Vector4::<Real>::new(-1_f64, 0_f64, 0_f64, 0_f64),
            DirectionEnum::YPos => Vector4::<Real>::new(0_f64, 1_f64, 0_f64, 0_f64),
            DirectionEnum::YNeg => Vector4::<Real>::new(0_f64, -1_f64, 0_f64, 0_f64),
            DirectionEnum::ZPos => Vector4::<Real>::new(0_f64, 0_f64, 1_f64, 0_f64),
            DirectionEnum::ZNeg => Vector4::<Real>::new(0_f64, 0_f64, -1_f64, 0_f64),
            DirectionEnum::TPos => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, 1_f64),
            DirectionEnum::TNeg => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, -1_f64),
        }
    }
    
    /// Get if the position is positive.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!( DirectionEnum::XPos.is_positive(), true);
    /// assert_eq!( DirectionEnum::TPos.is_positive(), true);
    /// assert_eq!( DirectionEnum::YNeg.is_positive(), false);
    /// ```
    pub const fn is_positive(self) -> bool {
        match self {
            DirectionEnum::XPos | DirectionEnum::YPos | DirectionEnum::ZPos | DirectionEnum::TPos => true,
            DirectionEnum::XNeg | DirectionEnum::YNeg | DirectionEnum::ZNeg | DirectionEnum::TNeg => false,
        }
    }
    
    /// Get if the position is Negative. see [`DirectionEnum::is_positive`]
    pub const fn is_negative(self) -> bool {
        ! self.is_positive()
    }
    
    /// Find the direction the vector point the most.
    /// For a zero vector return [`DirectionEnum::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// # use lattice_qcd_rs::dim::U4;
    /// # extern crate nalgebra;
    /// assert_eq!(DirectionEnum::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)), DirectionEnum::XPos);
    /// assert_eq!(DirectionEnum::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)), DirectionEnum::YNeg);
    /// assert_eq!(DirectionEnum::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)), DirectionEnum::TPos);
    /// ```
    pub fn from_vector(v: &Vector4<Real>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        // TODO try fold ?
        for i in 0..DirectionEnum::POSITIVES.len() {
            let scalar_prod = v.dot(&DirectionEnum::POSITIVES[i].to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        match index_max {
            0 => {
                if is_positive {
                    return DirectionEnum::XPos;
                }
                else {
                    return DirectionEnum::XNeg;
                }
            },
            1 => {
                if is_positive {
                    return DirectionEnum::YPos;
                }
                else {
                    return DirectionEnum::YNeg;
                }
            },
            2 => {
                if is_positive {
                    return DirectionEnum::ZPos;
                }
                else {
                    return DirectionEnum::ZNeg;
                }
            },
            3 => {
                if is_positive {
                    return DirectionEnum::TPos;
                }
                else {
                    return DirectionEnum::TNeg;
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
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XNeg.to_positive(), DirectionEnum::XPos);
    /// assert_eq!(DirectionEnum::YPos.to_positive(), DirectionEnum::YPos);
    /// ```
    pub const fn to_positive(self) -> Self {
        match self {
            DirectionEnum::XNeg => DirectionEnum::XPos,
            DirectionEnum::YNeg => DirectionEnum::YPos,
            DirectionEnum::ZNeg => DirectionEnum::ZPos,
            DirectionEnum::TNeg => DirectionEnum::TPos,
            _ => self,
        }
    }
    
    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XPos.to_index(), 0);
    /// assert_eq!(DirectionEnum::XNeg.to_index(), 0);
    /// assert_eq!(DirectionEnum::YPos.to_index(), 1);
    /// assert_eq!(DirectionEnum::YNeg.to_index(), 1);
    /// assert_eq!(DirectionEnum::ZPos.to_index(), 2);
    /// assert_eq!(DirectionEnum::ZNeg.to_index(), 2);
    /// assert_eq!(DirectionEnum::TPos.to_index(), 3);
    /// assert_eq!(DirectionEnum::TNeg.to_index(), 3);
    /// ```
    pub const fn to_index(self) -> usize {
        match self {
            DirectionEnum::XPos | DirectionEnum::XNeg => 0,
            DirectionEnum::YPos | DirectionEnum::YNeg => 1,
            DirectionEnum::ZPos | DirectionEnum::ZNeg => 2,
            DirectionEnum::TPos | DirectionEnum::TNeg => 3,
        }
    }
}

/// Return the negative of a direction
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::DirectionEnum;
/// assert_eq!(- DirectionEnum::XNeg, DirectionEnum::XPos);
/// assert_eq!(- DirectionEnum::YPos, DirectionEnum::YNeg);
/// ```
impl Neg for DirectionEnum {
    type Output = Self;
    
    fn neg(self) -> Self::Output {
        match self {
            DirectionEnum::XPos => DirectionEnum::XNeg,
            DirectionEnum::XNeg => DirectionEnum::XPos,
            DirectionEnum::YPos => DirectionEnum::YNeg,
            DirectionEnum::YNeg => DirectionEnum::YPos,
            DirectionEnum::ZPos => DirectionEnum::ZNeg,
            DirectionEnum::ZNeg => DirectionEnum::ZPos,
            DirectionEnum::TPos => DirectionEnum::TNeg,
            DirectionEnum::TNeg => DirectionEnum::TPos,
        }
    }
}

impl Neg for &DirectionEnum {
    type Output = Self;
    
    fn neg(self) -> Self::Output {
        match self {
            DirectionEnum::XPos => &DirectionEnum::XNeg,
            DirectionEnum::XNeg => &DirectionEnum::XPos,
            DirectionEnum::YPos => &DirectionEnum::YNeg,
            DirectionEnum::YNeg => &DirectionEnum::YPos,
            DirectionEnum::ZPos => &DirectionEnum::ZNeg,
            DirectionEnum::ZNeg => &DirectionEnum::ZPos,
            DirectionEnum::TPos => &DirectionEnum::TNeg,
            DirectionEnum::TNeg => &DirectionEnum::TPos,
        }
    }
}

/// Return [`DirectionEnum::to_index`].
impl From<DirectionEnum> for usize {
    fn from(d: DirectionEnum) -> Self {
        d.to_index()
    }
}

/// Return [`DirectionEnum::to_index`].
impl From<&DirectionEnum> for usize {
    fn from(d: &DirectionEnum) -> Self {
        d.to_index()
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<Vector4<Real>> for DirectionEnum {
    fn from(v: Vector4<Real>) -> Self {
        DirectionEnum::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<&Vector4<Real>> for DirectionEnum {
    fn from(v: &Vector4<Real>) -> Self {
        DirectionEnum::from_vector(v)
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<DirectionEnum> for Vector4<Real> {
    fn from(d: DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<&DirectionEnum> for Vector4<Real> {
    fn from(d: &DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}
