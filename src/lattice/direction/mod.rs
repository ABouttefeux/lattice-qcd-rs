//---------------------------------------
// mod

mod axis;
mod direction_enum;
mod oriented;

//---------------------------------------
// uses
use std::cmp::Ordering;
use std::convert::TryInto;
use std::error::Error;
use std::fmt::{self, Display};
use std::ops::Neg;

use lattice_qcd_rs_procedural_macro::{implement_direction_from, implement_direction_list};
use nalgebra::SVector;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::{Getter, Sealed};

pub use self::axis::Axis;
pub use self::direction_enum::DirectionEnum;
pub use self::oriented::OrientedDirection;
use super::{IndexToElement, LatticeCyclic, LatticeElementToIndex, NumberOfLatticeElement};
use crate::{private::Sealed, Real};

//---------------------------------------
// struct definition

/// Represent a cardinal direction
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy, Sealed)]
#[allow(clippy::unsafe_derive_deserialize)]
//^^^^^^ yes this is ok for this type,
// the use of unsafe is just there for some fnc to be constant which is crucial
// for performance. In the comment of [`Direction::directions_array`]
// it is explained why it is safe and therefore does not create any other value
// that safe code could create, so a manual implementation of [`serde::Deserialize`]
// is not necessary.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Getter)]
pub struct Direction<const D: usize> {
    /// Axis of the direction
    #[get(Pub, Const, copy)]
    #[get_mut(Pub)]
    axis: Axis<D>,
    /// Orientation
    // TODO fix doc about orientation
    is_positive: bool,
}

//---------------------------------------
// main impl block

impl<const D: usize> Direction<D> {
    /// New direction from a direction as an idex and wether it is in the positive direction.
    #[must_use]
    #[inline]
    pub const fn new(index_dir: usize, is_positive: bool) -> Option<Self> {
        // TODO return error ?
        if let Some(axis) = Axis::new(index_dir) {
            Some(Self { axis, is_positive })
        } else {
            None
        }
    }

    /// List of all positives directions.
    /// This is very slow use [`Self::positive_directions`] instead.
    #[allow(clippy::missing_panics_doc)] // it nevers panic
    #[must_use]
    #[inline]
    pub fn positives_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(D);
        for i in 0..D {
            x.push(Self::new(i, true).expect("unreachable"));
        }
        x
    }

    /// List all directions.
    /// This is very slow use [`DirectionList::directions`] instead.
    #[allow(clippy::missing_panics_doc)] // it nevers panic
    #[must_use]
    #[inline]
    pub fn directions_vec() -> Vec<Self> {
        let mut x = Vec::with_capacity(2 * D);
        for i in 0..D {
            x.push(Self::new(i, true).expect("unreachable"));
            x.push(Self::new(i, false).expect("unreachable"));
        }
        x
    }

    // TODO add const function for all direction once operation on const generic are added
    /// Get all direction with the sign `IS_POSITIVE`.
    #[must_use]
    #[inline]
    #[allow(unsafe_code)]
    pub const fn directions_array<const IS_POSITIVE: bool>() -> [Self; D] {
        // TODO use unsafe code to avoid the allocation
        let mut i = 0_usize;
        let mut array = [Self {
            // SAFETY: if D = 0 the array is empty so no axis is created
            axis: unsafe { Axis::new_unchecked(0) },
            is_positive: IS_POSITIVE,
        }; D];
        while i < D {
            array[i] = Self {
                // SAFETY: i < D
                axis: unsafe { Axis::new_unchecked(i) },
                is_positive: IS_POSITIVE,
            };
            i += 1;
        }
        array
    }

    /// Get all negative direction
    #[must_use]
    #[inline]
    pub const fn negative_directions() -> [Self; D] {
        Self::directions_array::<false>()
    }

    /// Get all positive direction
    #[allow(clippy::same_name_method)]
    #[must_use]
    #[inline]
    pub const fn positive_directions() -> [Self; D] {
        Self::directions_array::<true>()
    }

    /// Get if the orientation is positive.
    #[must_use]
    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.is_positive
    }

    /// Get if the orientation is negative, see [`Self::is_positive`].
    #[must_use]
    #[inline]
    pub const fn is_negative(&self) -> bool {
        !self.is_positive()
    }

    /// Return the direction with positive orientation, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{lattice::Direction, error::ImplementationError};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert_eq!(
    ///     Direction::<4>::new(1, false)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .to_positive(),
    ///     Direction::<4>::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     Direction::<4>::new(1, true)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .to_positive(),
    ///     Direction::<4>::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn to_positive(mut self) -> Self {
        self.is_positive = true;
        self
    }

    /// Get a index associated to the axis of the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{lattice::Direction, error::ImplementationError};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert_eq!(
    ///     Direction::<4>::new(1, false)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     1
    /// );
    /// assert_eq!(
    ///     Direction::<4>::new(3, true)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     3
    /// );
    /// assert_eq!(
    ///     Direction::<6>::new(5, false)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     5
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn index(&self) -> usize {
        self.axis().index()
    }

    /// Convert the direction into a vector of norm `a`;
    #[must_use]
    #[inline]
    pub fn to_vector(self, a: f64) -> SVector<Real, D> {
        self.to_unit_vector() * a
    }

    /// Returns the dimension.
    #[must_use]
    #[inline]
    pub const fn dim() -> usize {
        D
    }

    /// Convert the direction into a vector of norm `1`
    // TODO example
    #[must_use]
    #[inline]
    pub fn to_unit_vector(self) -> SVector<Real, D> {
        let mut v = SVector::zeros();
        v[self.index()] = 1_f64;
        v
    }

    /// Find the direction the vector point the most.
    /// For a zero vector return [`DirectionEnum::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::Direction;
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box< dyn std::error::Error>> {
    /// assert_eq!(
    ///     Direction::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)),
    ///     Direction::<4>::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     Direction::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)),
    ///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     Direction::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)),
    ///     Direction::<4>::new(3, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::missing_panics_doc)] // it nevers panic
    #[must_use]
    #[inline]
    pub fn from_vector(v: &SVector<Real, D>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for (i, dir) in Self::positive_directions().iter().enumerate() {
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

//---------------------------------------
// Common trait implementation

impl<const D: usize> Display for Direction<D> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[index {}, is positive {}]",
            self.index(),
            self.is_positive()
        )
    }
}

/// Partial ordering is set as follows: two directions can be compared if they have the same index
/// or the same direction sign. In the first case a positive direction is greater than a negative direction
/// In the latter case the ordering is done on the index.
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::Direction;
/// # use lattice_qcd_rs::error::ImplementationError;
/// use std::cmp::Ordering;
/// # fn main() -> Result<(), Box< dyn std::error::Error>> {
///
/// let dir_1 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_2 =
///     Direction::<4>::new(1, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert!(dir_1 < dir_2);
/// assert_eq!(dir_1.partial_cmp(&dir_2), Some(Ordering::Less));
/// //--------
/// let dir_3 =
///     Direction::<4>::new(3, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_4 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert!(dir_3 > dir_4);
/// assert_eq!(dir_3.partial_cmp(&dir_4), Some(Ordering::Greater));
/// //--------
/// let dir_5 =
///     Direction::<4>::new(3, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_6 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert_eq!(dir_5.partial_cmp(&dir_6), None);
/// //--------
/// let dir_5 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// let dir_6 =
///     Direction::<4>::new(1, false).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
/// assert_eq!(dir_5.partial_cmp(&dir_6), Some(Ordering::Equal));
/// # Ok(())
/// # }
/// ```
impl<const D: usize> PartialOrd for Direction<D> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.is_positive() == other.is_positive() {
            self.index().partial_cmp(&other.index())
        } else if self.index() == other.index() {
            self.is_positive().partial_cmp(&other.is_positive())
        } else {
            None
        }
    }
}

//---------------------------------------
// Ops traits

impl<const D: usize> Neg for Direction<D> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self.is_positive = !self.is_positive;
        self
    }
}

impl<const D: usize> Neg for &Direction<D> {
    type Output = Direction<D>;

    #[inline]
    fn neg(self) -> Self::Output {
        -*self
    }
}

//---------------------------------------
// Conversion

/// Return [`Direction::index`].
impl<const D: usize> From<Direction<D>> for usize {
    #[inline]
    fn from(d: Direction<D>) -> Self {
        d.index()
    }
}

/// Return [`Direction::index`].
impl<const D: usize> From<&Direction<D>> for usize {
    #[inline]
    fn from(d: &Direction<D>) -> Self {
        d.index()
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<const D: usize> From<SVector<Real, D>> for Direction<D> {
    #[inline]
    fn from(v: SVector<Real, D>) -> Self {
        Self::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl<const D: usize> From<&SVector<Real, D>> for Direction<D> {
    #[inline]
    fn from(v: &SVector<Real, D>) -> Self {
        Self::from_vector(v)
    }
}

/// Return [`Direction::to_unit_vector`].
impl<const D: usize> From<Direction<D>> for SVector<Real, D> {
    #[inline]
    fn from(d: Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`Direction::to_unit_vector`].
impl<const D: usize> From<&Direction<D>> for SVector<Real, D> {
    #[inline]
    fn from(d: &Direction<D>) -> Self {
        d.to_unit_vector()
    }
}

impl From<DirectionEnum> for Direction<4> {
    #[inline]
    fn from(d: DirectionEnum) -> Self {
        Self::new(DirectionEnum::index(d), d.is_positive()).expect("unreachable")
    }
}

impl From<Direction<4>> for DirectionEnum {
    #[inline]
    fn from(d: Direction<4>) -> Self {
        d.into()
    }
}

impl From<&Direction<4>> for DirectionEnum {
    #[inline]
    fn from(d: &Direction<4>) -> Self {
        match (d.index(), d.is_positive()) {
            (0, true) => Self::XPos,
            (0, false) => Self::XNeg,
            (1, true) => Self::YPos,
            (1, false) => Self::YNeg,
            (2, true) => Self::ZPos,
            (2, false) => Self::ZNeg,
            (3, true) => Self::TPos,
            (3, false) => Self::TNeg,
            _ => unreachable!(),
        }
    }
}

impl From<&DirectionEnum> for Direction<4> {
    #[inline]
    fn from(d: &DirectionEnum) -> Self {
        Self::new(DirectionEnum::index(*d), d.is_positive()).expect("unreachable")
    }
}

// with Direction

/// create a [`Direction`] with positive orientation of the same axis.
impl<const D: usize> From<Axis<D>> for Direction<D> {
    #[inline]
    fn from(value: Axis<D>) -> Self {
        Self::new(value.index(), true).expect("always exists")
    }
}

impl<const D: usize> From<Direction<D>> for Axis<D> {
    #[inline]
    fn from(value: Direction<D>) -> Self {
        value.axis()
    }
}

impl<const D: usize> AsRef<Axis<D>> for Direction<D> {
    #[inline]
    fn as_ref(&self) -> &Axis<D> {
        &self.axis
    }
}

impl<const D: usize> AsMut<Axis<D>> for Direction<D> {
    #[inline]
    fn as_mut(&mut self) -> &mut Axis<D> {
        self.axis_mut()
    }
}

//---------------------------------------
// Procedural macro implementation

/// List all possible direction
pub trait DirectionList: Sized {
    /// List all directions.
    #[must_use]
    fn directions() -> &'static [Self];
    /// List all positive directions.
    #[must_use]
    fn positive_directions() -> &'static [Self];
}

/// Error return by [`TryFrom`] for [`Direction`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum DirectionConversionError {
    /// The index is out of bound, i.e. the direction axis does not exist in the lower space dimension.
    IndexOutOfBound, // more error info like dim and index
}

impl Display for DirectionConversionError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBound => write!(f, "the index is out of bound, the direction axis does not exist in the lower space dimension"),
        }
    }
}

impl Error for DirectionConversionError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IndexOutOfBound => None,
        }
    }
}

implement_direction_list!();

implement_direction_from!();

//---------------------------------------
// trait DirectionIndexing

/// An internal trait that direction implements. It is used for auto implementation
/// of trait to avoid conflict. For example [`LatticeElementToIndex`] is implemented
/// for type that are [`DirectionIndexing`] and [`DirectionTrait`]. Moreover we want
///
///
/// This trait is a super trait of [`Sealed`] which is private meaning that It can't be
/// implemented outside of this trait.
pub trait DirectionTrait: Sealed {}

/// Trait to transform (directions) Types to and from indices independently of a Lattice
/// Contrary to [`LatticeElementToIndex`] [`NumberOfLatticeElement`] and [`IndexToElement`].
///
/// This trait is used to automate the code generation for [`Iterator`] and
/// [`rayon::iter::IndexedParallelIterator`] without the overhead of a lattice.
///
///
/// This trait is a super trait of [`Sealed`] which is private meaning that i
/// it can't be implemented outside of this trait.
pub trait DirectionIndexing: Sealed + Sized {
    /// Transform an element to an index.
    #[must_use]
    fn direction_to_index(&self) -> usize;

    /// Convert an element to an index.
    #[must_use]
    fn direction_from_index(index: usize) -> Option<Self>;

    /// Determine the number of element possible.
    #[must_use]
    fn number_of_directions() -> usize;
}

//---------------------------------------
// trait auto impl

impl<const D: usize, T: DirectionIndexing + DirectionTrait> LatticeElementToIndex<D> for T {
    #[inline]
    fn to_index(&self, _lattice: &LatticeCyclic<D>) -> usize {
        self.direction_to_index()
    }
}

impl<const D: usize, T: DirectionIndexing + DirectionTrait> NumberOfLatticeElement<D> for T {
    #[inline]
    fn number_of_elements(_lattice: &LatticeCyclic<D>) -> usize {
        T::number_of_directions()
    }
}

impl<const D: usize, T: DirectionIndexing + DirectionTrait> IndexToElement<D> for T {
    #[inline]
    fn index_to_element(_lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
        T::direction_from_index(index)
    }
}

//---------------------------------------
// direction trait impl

impl<const D: usize> DirectionTrait for Direction<D> {}

impl<const D: usize> DirectionIndexing for Direction<D> {
    #[inline]
    fn direction_to_index(&self) -> usize {
        self.index() * 2 + usize::from(!self.is_positive())
    }

    #[inline]
    fn direction_from_index(index: usize) -> Option<Self> {
        Self::new((index.saturating_sub(1)) / 2, index % 2 == 0)
    }

    #[inline]
    fn number_of_directions() -> usize {
        D * 2
    }
}
