//---------------------------------------
// uses

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};
use utils_lib::{Getter, Sealed};

use super::DirectionIndexing;

//---------------------------------------
// struct definition

/// Represent an axis in the space. There is `D` axis in dimension D. Contrary to
/// [`super::Direction`] and [`super::OrientedDirection`], an [`Axis`]
/// does not have an orientation.
#[derive(Sealed, Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Getter)]
pub struct Axis<const D: usize> {
    /// index of the axis
    #[get(Pub, Const, Copy, self_ty = "value")]
    index: usize,
}

//---------------------------------------
// main impl block

impl<const D: usize> Axis<D> {
    /// Create a new axis. The index should be strictly smaller than `D` to return [`Some`].
    /// # Example
    /// ```
    /// use lattice_qcd_rs::{error::ImplementationError, lattice::Axis};
    /// # fn main() -> Result<(), ImplementationError> {
    /// assert!(Axis::<0>::new(0).is_none());
    /// assert!(Axis::<0>::new(4).is_none());
    ///
    /// let axis = Axis::<1>::new(0).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(axis.index(), 0);
    /// assert!(Axis::<1>::new(1).is_none());
    ///
    /// let axis = Axis::<4>::new(3).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// assert_eq!(axis.index(), 3);
    /// assert!(Axis::<4>::new(4).is_none());
    /// assert!(Axis::<4>::new(6).is_none());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[allow(clippy::if_then_some_else_none)] // not possible for const fn
    pub const fn new(index: usize) -> Option<Self> {
        if index < D {
            Some(Self { index })
        } else {
            None
        }
    }

    #[inline]
    #[must_use]
    #[allow(unsafe_code)]
    pub(super) const unsafe fn new_unchecked(index: usize) -> Self {
        Self { index }
    }
}

//---------------------------------------
// conversion

// with usize

impl<const D: usize> From<Axis<D>> for usize {
    #[inline]
    fn from(value: Axis<D>) -> Self {
        value.index()
    }
}

impl<const D: usize> AsRef<usize> for Axis<D> {
    #[inline]
    fn as_ref(&self) -> &usize {
        &self.index
    }
}

// there is no AsMut as is it not safe to give a mut ref to the inner index

//---------------------------------------
// lattice indexing

impl<const D: usize> DirectionIndexing for Axis<D> {
    #[inline]
    fn to_index(&self) -> usize {
        self.index()
    }

    #[inline]
    fn from_index(index: usize) -> Option<Self> {
        Self::new(index)
    }

    #[inline]
    fn number_of_elements() -> usize {
        D
    }
}

#[cfg(test)]
mod test {
    use crate::{error::ImplementationError, lattice::Axis};

    #[test]
    fn axis() -> Result<(), ImplementationError> {
        for i in 0..25 {
            assert!(Axis::<0>::new(i).is_none());
        }
        //------------------
        assert_eq!(
            Axis::<1>::new(0).ok_or(ImplementationError::OptionWithUnexpectedNone)?,
            Axis { index: 0 }
        );
        for i in 1..25 {
            assert!(Axis::<1>::new(i).is_none());
        }
        //------------------
        for i in 0..4 {
            assert_eq!(
                Axis::<4>::new(i).ok_or(ImplementationError::OptionWithUnexpectedNone)?,
                Axis { index: i }
            );
        }
        for i in 4..25 {
            assert!(Axis::<4>::new(i).is_none());
        }

        Ok(())
    }
}
