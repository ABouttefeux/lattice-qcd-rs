//---------------------------------------
// uses

use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::ops::Neg;

use nalgebra::Vector4;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::Sealed;

use super::{Direction, DirectionList};
use crate::lattice::{LatticeCyclic, LatticeElementToIndex};
use crate::Real;

//---------------------------------------
// struct definition

// TODO depreciate ?
/// Represent a cardinal direction
#[allow(clippy::exhaustive_enums)]
#[derive(Sealed, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum DirectionEnum {
    /// Positive x direction.
    XPos,
    /// Negative x direction.
    XNeg,
    /// Positive y direction.
    YPos,
    /// Negative y direction.
    YNeg,
    /// Positive z direction.
    ZPos,
    /// Negative z direction.
    ZNeg,
    /// Positive t direction.
    TPos,
    /// Negative t direction.
    TNeg,
}

//---------------------------------------
// main impl block

impl DirectionEnum {
    /// List all directions.
    pub const DIRECTIONS: [Self; 8] = [
        Self::XPos,
        Self::YPos,
        Self::ZPos,
        Self::TPos,
        Self::XNeg,
        Self::YNeg,
        Self::ZNeg,
        Self::TNeg,
    ];
    /// List all spatial directions.
    pub const DIRECTIONS_SPACE: [Self; 6] = [
        Self::XPos,
        Self::YPos,
        Self::ZPos,
        Self::XNeg,
        Self::YNeg,
        Self::ZNeg,
    ];
    /// List of all positives directions.
    pub const POSITIVES: [Self; 4] = [Self::XPos, Self::YPos, Self::ZPos, Self::TPos];
    /// List spatial positive direction.
    pub const POSITIVES_SPACE: [Self; 3] = [Self::XPos, Self::YPos, Self::ZPos];

    /// Convert the direction into a vector of norm `a`;
    #[must_use]
    #[inline]
    pub fn to_vector(self, a: f64) -> Vector4<Real> {
        self.to_unit_vector() * a
    }

    /// Convert the direction into a vector of norm `1`;
    #[must_use]
    #[inline]
    pub const fn to_unit_vector(self) -> Vector4<Real> {
        match self {
            Self::XPos => Vector4::<Real>::new(1_f64, 0_f64, 0_f64, 0_f64),
            Self::XNeg => Vector4::<Real>::new(-1_f64, 0_f64, 0_f64, 0_f64),
            Self::YPos => Vector4::<Real>::new(0_f64, 1_f64, 0_f64, 0_f64),
            Self::YNeg => Vector4::<Real>::new(0_f64, -1_f64, 0_f64, 0_f64),
            Self::ZPos => Vector4::<Real>::new(0_f64, 0_f64, 1_f64, 0_f64),
            Self::ZNeg => Vector4::<Real>::new(0_f64, 0_f64, -1_f64, 0_f64),
            Self::TPos => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, 1_f64),
            Self::TNeg => Vector4::<Real>::new(0_f64, 0_f64, 0_f64, -1_f64),
        }
    }

    /// Get if the position is positive.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XPos.is_positive(), true);
    /// assert_eq!(DirectionEnum::TPos.is_positive(), true);
    /// assert_eq!(DirectionEnum::YNeg.is_positive(), false);
    /// ```
    #[must_use]
    #[inline]
    pub const fn is_positive(self) -> bool {
        match self {
            Self::XPos | Self::YPos | Self::ZPos | Self::TPos => true,
            Self::XNeg | Self::YNeg | Self::ZNeg | Self::TNeg => false,
        }
    }

    /// Get if the position is Negative. see [`DirectionEnum::is_positive`]
    #[must_use]
    #[inline]
    pub const fn is_negative(self) -> bool {
        !self.is_positive()
    }

    /// Find the direction the vector point the most.
    /// For a zero vector return [`DirectionEnum::XPos`].
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// # extern crate nalgebra;
    /// assert_eq!(
    ///     DirectionEnum::from_vector(&nalgebra::Vector4::new(1_f64, 0_f64, 0_f64, 0_f64)),
    ///     DirectionEnum::XPos
    /// );
    /// assert_eq!(
    ///     DirectionEnum::from_vector(&nalgebra::Vector4::new(0_f64, -1_f64, 0_f64, 0_f64)),
    ///     DirectionEnum::YNeg
    /// );
    /// assert_eq!(
    ///     DirectionEnum::from_vector(&nalgebra::Vector4::new(0.5_f64, 1_f64, 0_f64, 2_f64)),
    ///     DirectionEnum::TPos
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_vector(v: &Vector4<Real>) -> Self {
        let mut max = 0_f64;
        let mut index_max: usize = 0;
        let mut is_positive = true;
        for i in 0..Self::POSITIVES.len() {
            let scalar_prod = v.dot(&Self::POSITIVES[i].to_vector(1_f64));
            if scalar_prod.abs() > max {
                max = scalar_prod.abs();
                index_max = i;
                is_positive = scalar_prod > 0_f64;
            }
        }
        match index_max {
            0 => {
                if is_positive {
                    Self::XPos
                } else {
                    Self::XNeg
                }
            }
            1 => {
                if is_positive {
                    Self::YPos
                } else {
                    Self::YNeg
                }
            }
            2 => {
                if is_positive {
                    Self::ZPos
                } else {
                    Self::ZNeg
                }
            }
            3 => {
                if is_positive {
                    Self::TPos
                } else {
                    Self::TNeg
                }
            }
            _ => {
                // the code should attain this code. and therefore panicking is not expected.
                unreachable!("Implementation error : invalid index")
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
    #[inline]
    #[must_use]
    pub const fn to_positive(self) -> Self {
        match self {
            Self::XNeg => Self::XPos,
            Self::YNeg => Self::YPos,
            Self::ZNeg => Self::ZPos,
            Self::TNeg => Self::TPos,
            _ => self,
        }
    }

    /// Get a index associated to the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::DirectionEnum;
    /// assert_eq!(DirectionEnum::XPos.index(), 0);
    /// assert_eq!(DirectionEnum::XNeg.index(), 0);
    /// assert_eq!(DirectionEnum::YPos.index(), 1);
    /// assert_eq!(DirectionEnum::YNeg.index(), 1);
    /// assert_eq!(DirectionEnum::ZPos.index(), 2);
    /// assert_eq!(DirectionEnum::ZNeg.index(), 2);
    /// assert_eq!(DirectionEnum::TPos.index(), 3);
    /// assert_eq!(DirectionEnum::TNeg.index(), 3);
    /// ```
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::XPos | Self::XNeg => 0,
            Self::YPos | Self::YNeg => 1,
            Self::ZPos | Self::ZNeg => 2,
            Self::TPos | Self::TNeg => 3,
        }
    }
}

//---------------------------------------
// Common traits

/// Return [`DirectionEnum::XPos`]
impl Default for DirectionEnum {
    ///Return [`DirectionEnum::XPos`]
    #[inline]
    fn default() -> Self {
        Self::XPos
    }
}

impl Display for DirectionEnum {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::XPos => write!(f, "positive X direction"),
            Self::XNeg => write!(f, "negative X direction"),
            Self::YPos => write!(f, "positive Y direction"),
            Self::YNeg => write!(f, "negative Y direction"),
            Self::ZPos => write!(f, "positive Z direction"),
            Self::ZNeg => write!(f, "negative Z direction"),
            Self::TPos => write!(f, "positive T direction"),
            Self::TNeg => write!(f, "negative T direction"),
        }
    }
}

impl PartialOrd for DirectionEnum {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Direction::<4>::from(self).partial_cmp(&other.into())
    }
}

//---------------------------------------
// DirectionList trait

impl DirectionList for DirectionEnum {
    #[inline]
    fn directions() -> &'static [Self] {
        &Self::DIRECTIONS
    }

    #[inline]
    fn positive_directions() -> &'static [Self] {
        &Self::POSITIVES
    }
}

//---------------------------------------
// Ops trait

/// Return the negative of a direction
/// # Example
/// ```
/// # use lattice_qcd_rs::lattice::DirectionEnum;
/// assert_eq!(-DirectionEnum::XNeg, DirectionEnum::XPos);
/// assert_eq!(-DirectionEnum::YPos, DirectionEnum::YNeg);
/// ```
impl Neg for DirectionEnum {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Self::XPos => Self::XNeg,
            Self::XNeg => Self::XPos,
            Self::YPos => Self::YNeg,
            Self::YNeg => Self::YPos,
            Self::ZPos => Self::ZNeg,
            Self::ZNeg => Self::ZPos,
            Self::TPos => Self::TNeg,
            Self::TNeg => Self::TPos,
        }
    }
}

impl Neg for &DirectionEnum {
    type Output = Self;

    #[inline]
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

//---------------------------------------
// Conversion

/// Return [`DirectionEnum::to_index`].
impl From<DirectionEnum> for usize {
    #[inline]
    fn from(d: DirectionEnum) -> Self {
        d.index()
    }
}

/// Return [`DirectionEnum::to_index`].
impl From<&DirectionEnum> for usize {
    #[inline]
    fn from(d: &DirectionEnum) -> Self {
        DirectionEnum::index(*d)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<Vector4<Real>> for DirectionEnum {
    #[inline]
    fn from(v: Vector4<Real>) -> Self {
        Self::from_vector(&v)
    }
}

/// Return [`DirectionEnum::from_vector`].
impl From<&Vector4<Real>> for DirectionEnum {
    #[inline]
    fn from(v: &Vector4<Real>) -> Self {
        Self::from_vector(v)
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<DirectionEnum> for Vector4<Real> {
    #[inline]
    fn from(d: DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

/// Return [`DirectionEnum::to_unit_vector`].
impl From<&DirectionEnum> for Vector4<Real> {
    #[inline]
    fn from(d: &DirectionEnum) -> Self {
        d.to_unit_vector()
    }
}

//---------------------------------------
// Lattice index traits

impl LatticeElementToIndex<4> for DirectionEnum {
    #[inline]
    fn to_index(&self, l: &LatticeCyclic<4>) -> usize {
        Direction::<4>::from(self).to_index(l)
    }
}

// impl NumberOfLatticeElement<4> for DirectionEnum {
//     #[inline]
//     fn number_of_elements(lattice: &LatticeCyclic<4>) -> usize {
//         Direction::<4>::number_of_elements(lattice)
//     }
// }

// impl IndexToElement<4> for DirectionEnum {
//     fn index_to_element(lattice: &LatticeCyclic<4>, index: usize) -> Option<Self> {
//         Direction::<4>::index_to_element(lattice, index).map(Into::into)
//     }
// }
