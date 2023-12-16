//! Contains [`OrientedDirection`]

//---------------------------------------
// uses

use std::{
    error::Error,
    fmt::{self, Display},
    ops::Neg,
    usize,
};

use nalgebra::SVector;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::{Getter, Sealed};

use super::{Axis, Direction, DirectionIndexing, DirectionTrait};
use crate::Real;

//---------------------------------------
// struct definition

/// A cardinal direction whose orientation is set in the type.
/// It is similar to [`Direction`] and to [`Axis`].
#[derive(Sealed, Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Getter)]
pub struct OrientedDirection<const D: usize, const ORIENTATION: bool> {
    /// axis of the direction
    #[get(Pub, Const, Copy, self_ty = "value")]
    #[get_mut(Pub)]
    axis: Axis<D>,
}

//---------------------------------------
// main impl block

impl<const D: usize, const ORIENTATION: bool> OrientedDirection<D, ORIENTATION> {
    /// Create a new oriented direction. It returns [`Some`] only if `index < D`
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{error::ImplementationError, lattice::OrientedDirection};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert!(OrientedDirection::<0, true>::new(0).is_none());
    /// assert!(OrientedDirection::<0, false>::new(4).is_none());
    ///
    /// let o_dir = OrientedDirection::<1, true>::new(0)
    ///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(o_dir.index(), 0);
    /// assert!(OrientedDirection::<1, false>::new(1).is_none());
    ///
    /// let o_dir = OrientedDirection::<4, true>::new(3)
    ///     .ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(o_dir.index(), 3);
    /// assert!(OrientedDirection::<4, false>::new(4).is_none());
    /// assert!(OrientedDirection::<4, true>::new(6).is_none());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(index: usize) -> Option<Self> {
        if let Some(axis) = Axis::new(index) {
            Some(Self { axis })
        } else {
            None
        }
    }

    /// Get a index associated to the axis of the direction.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{lattice::OrientedDirection, error::ImplementationError};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert_eq!(
    ///     OrientedDirection::<4, false>::new(1)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     1
    /// );
    /// assert_eq!(
    ///     OrientedDirection::<4, true>::new(3)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     3
    /// );
    /// assert_eq!(
    ///     OrientedDirection::<6, false>::new(5)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .index(),
    ///     5
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.axis().index()
    }

    /// Get if the orientation is positive.
    #[inline]
    #[must_use]
    pub const fn is_positive() -> bool {
        ORIENTATION
    }

    /// Get if the orientation is negative, see [`Self::is_positive`].
    #[inline]
    #[must_use]
    pub const fn is_negative() -> bool {
        !ORIENTATION
    }

    /// Returns the dimension.
    #[inline]
    #[must_use]
    pub const fn dim() -> usize {
        D
    }

    /// Return the cardinal direction with positive orientation, for example `-x` gives `+x`
    /// and `+x` gives `+x`.
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{error::ImplementationError, lattice::OrientedDirection};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert_eq!(
    ///     OrientedDirection::<4, false>::new(1)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .to_positive(),
    ///     OrientedDirection::<4, true>::new(1)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     OrientedDirection::<4, true>::new(1)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    ///         .to_positive(),
    ///     OrientedDirection::<4, true>::new(1)
    ///         .ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn to_positive(self) -> OrientedDirection<D, true> {
        // TODO use expect when it is constant
        if let Some(c) = OrientedDirection::new(self.index()) {
            c
        } else {
            unreachable!()
        }
    }

    /// Convert the direction into a vector of norm `1`;
    #[must_use]
    #[inline]
    pub fn to_unit_vector(self) -> SVector<Real, D> {
        Direction::from(self).to_unit_vector()
    }
}

//---------------------------------------
// conversion

// with self

impl<const D: usize> From<OrientedDirection<D, true>> for OrientedDirection<D, false> {
    #[inline]
    fn from(value: OrientedDirection<D, true>) -> Self {
        Self::new(value.axis().index()).expect("always exist")
    }
}

impl<const D: usize> From<OrientedDirection<D, false>> for OrientedDirection<D, true> {
    #[inline]
    fn from(value: OrientedDirection<D, false>) -> Self {
        Self::new(value.axis().index()).expect("always exist")
    }
}

// with axis

impl<const D: usize, const DIR: bool> From<Axis<D>> for OrientedDirection<D, DIR> {
    #[inline]
    fn from(value: Axis<D>) -> Self {
        Self::new(value.index()).expect("always exists")
    }
}

impl<const D: usize, const DIR: bool> From<OrientedDirection<D, DIR>> for Axis<D> {
    #[inline]
    fn from(value: OrientedDirection<D, DIR>) -> Self {
        value.axis()
    }
}

impl<const D: usize, const DIR: bool> AsRef<Axis<D>> for OrientedDirection<D, DIR> {
    #[inline]
    fn as_ref(&self) -> &Axis<D> {
        &self.axis
    }
}

impl<const D: usize, const DIR: bool> AsMut<Axis<D>> for OrientedDirection<D, DIR> {
    #[inline]
    fn as_mut(&mut self) -> &mut Axis<D> {
        &mut self.axis
    }
}

// with direction

impl<const D: usize, const DIR: bool> From<OrientedDirection<D, DIR>> for Direction<D> {
    #[inline]
    fn from(value: OrientedDirection<D, DIR>) -> Self {
        Self::new(value.index(), DIR).expect("always exist")
    }
}

/// Error while converting a [`Direction`] into an [`OrientedDirection`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum OrientedDirectionConversionError {
    /// The orientation does not correspond to the type of the oriented direction.
    WrongOrientation,
    /// The index is out of bound.
    IndexOutOfBound,
}

impl Display for OrientedDirectionConversionError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conversion error: ")?;
        match self {
            Self::WrongOrientation => write!(
                f,
                "the orientation does not correspond to the type of the oriented direction"
            ),
            Self::IndexOutOfBound => write!(f, "the index is out of bound"),
        }
    }
}

impl Error for OrientedDirectionConversionError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::WrongOrientation | Self::IndexOutOfBound => None,
        }
    }
}

impl<const D: usize, const DIR: bool> TryFrom<Direction<D>> for OrientedDirection<D, DIR> {
    type Error = OrientedDirectionConversionError;

    #[inline]
    fn try_from(dir: Direction<D>) -> Result<Self, Self::Error> {
        if dir.is_positive() && DIR {
            Self::new(dir.index()).ok_or(OrientedDirectionConversionError::IndexOutOfBound)
        } else {
            Err(OrientedDirectionConversionError::WrongOrientation)
        }
    }
}

// with usize

impl<const D: usize, const DIR: bool> From<OrientedDirection<D, DIR>> for usize {
    #[inline]
    fn from(value: OrientedDirection<D, DIR>) -> Self {
        value.index()
    }
}

impl<const D: usize, const DIR: bool> AsRef<usize> for OrientedDirection<D, DIR> {
    #[inline]
    fn as_ref(&self) -> &usize {
        <Self as AsRef<Axis<D>>>::as_ref(self).as_ref()
    }
}

//---------------------------------------
// operator

impl<const D: usize> Neg for OrientedDirection<D, true> {
    type Output = OrientedDirection<D, false>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.into()
    }
}

impl<const D: usize> Neg for OrientedDirection<D, false> {
    type Output = OrientedDirection<D, true>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.into()
    }
}

//---------------------------------------
// lattice indexing

impl<const D: usize, const ORIENTATION: bool> DirectionIndexing
    for OrientedDirection<D, ORIENTATION>
{
    #[inline]
    fn direction_to_index(&self) -> usize {
        self.index()
    }

    #[inline]
    fn direction_from_index(index: usize) -> Option<Self> {
        Self::new(index)
    }

    #[inline]
    fn number_of_directions() -> usize {
        D
    }
}

impl<const D: usize, const ORIENTATION: bool> DirectionTrait for OrientedDirection<D, ORIENTATION> {}

// /// TODO impl doc
// impl<const D: usize> NumberOfLatticeElement<D> for Direction<D> {
//     #[inline]
//     fn number_of_elements(_lattice: &LatticeCyclic<D>) -> usize {
//         D
//     }
// }

// // TODO impl doc
// impl<const D: usize> IndexToElement<D> for Direction<D> {
//     fn index_to_element(_lattice: &LatticeCyclic<D>, index: usize) -> Option<Self> {
//         Self::new(index, true)
//     }
// }
