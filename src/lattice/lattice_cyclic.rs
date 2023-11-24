use std::fmt::{self, Display};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::{
    Direction, IteratorLatticeLinkCanonical, IteratorLatticePoint, LatticeLink,
    LatticeLinkCanonical, LatticePoint,
};
use crate::{
    error::LatticeInitializationError,
    prelude::{EField, LinkMatrix},
    Real,
};

/// A Cyclic lattice in space. Does not store point and links but is used to generate them.
///
/// The generic parameter `D` is the dimension.
///
/// This lattice is Cyclic more precisely if the lattice has N points in each direction.
/// Then we can move alongside a direction going though point 0, 1, ... N-1. The next step in
/// the same direction goes back to the point at 0.
///
/// This structure is used for operation on [`LatticePoint`], [`LatticeLink`] and
/// [`LatticeLinkCanonical`].
// For example, theses three structures are abstract and are in general use to
/// access data on the lattice. These data are stored [`LinkMatrix`] and [`EField`] which are just
/// a wrapper around a [`Vec`]. `LatticeCyclic` is used to convert the lattice element to
/// an index to access these data.
///
/// This contain very few data and can be cloned at almost no cost even though
/// it does not implement [`Copy`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct LatticeCyclic<const D: usize> {
    /// The lattice spacing.
    size: Real,
    /// The number of point *per* dimension.
    dim: usize,
}

impl<const D: usize> LatticeCyclic<D> {
    /// Number space + time dimension, this is the `D` parameter.
    ///
    /// Not to confuse with [`LatticeCyclic::dim`] which is the number of point per dimension.
    #[must_use]
    #[inline]
    pub const fn dim_st() -> usize {
        D
    }

    /// see [`LatticeLinkCanonical`], a conical link is a link whose direction is always positive.
    /// That means that a link form `[x, y, z, t]` with direction `-x`
    /// the link return is `[x - 1, y, z, t]` (modulo the `lattice::dim()`) with direction `+x`
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclic, DirectionEnum, LatticePoint, LatticeLinkCanonical};
    /// # use lattice_qcd_rs::error::ImplementationError;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let point = LatticePoint::<4>::new([1, 0, 2, 0].into());
    /// assert_eq!(
    ///     lattice.link_canonical(point, DirectionEnum::XNeg.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([0, 0, 2, 0].into()), DirectionEnum::XPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     lattice.link_canonical(point, DirectionEnum::XPos.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([1, 0, 2, 0].into()), DirectionEnum::XPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// assert_eq!(
    ///     lattice.link_canonical(point, DirectionEnum::YNeg.into()),
    ///     LatticeLinkCanonical::new(LatticePoint::new([1, 3, 2, 0].into()), DirectionEnum::YPos.into()).ok_or(ImplementationError::OptionWithUnexpectedNone)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::missing_panics_doc)] // does not panic
    #[must_use]
    #[inline]
    pub fn link_canonical(
        &self,
        pos: LatticePoint<D>,
        dir: Direction<D>,
    ) -> LatticeLinkCanonical<D> {
        let mut pos_link = pos;
        if !dir.is_positive() {
            pos_link = self.add_point_direction(pos_link, &dir);
        }
        for i in 0..pos.len() {
            pos_link[i] %= self.dim();
        }
        LatticeLinkCanonical::new(pos_link, dir.to_positive()).expect("exist")
    }

    /// Return a link build from `pos` and `dir`.
    ///
    /// It is similar to [`LatticeLink::new`]. It however enforce that the point is inside the bounds.
    /// If it is not, it will use the modulus of the bound.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::{lattice::{LatticeCyclic, Direction, LatticePoint}, error::ImplementationError};
    /// # use nalgebra::SVector;
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let l = LatticeCyclic::<3>::new(1_f64, 4)?;
    /// let dir = Direction::new(0, true).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let pt = LatticePoint::new(SVector::<_, 3>::new(1, 2, 5));
    /// let link = l.link(pt, dir);
    /// assert_eq!(
    ///     *link.pos(),
    ///     LatticePoint::new(SVector::<_, 3>::new(1, 2, 1))
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn link(&self, pos: LatticePoint<D>, dir: Direction<D>) -> LatticeLink<D> {
        let mut pos_link = LatticePoint::new_zero();
        for i in 0..pos.len() {
            pos_link[i] = pos[i] % self.dim();
        }
        LatticeLink::new(pos_link, dir)
    }

    /// Transform a [`LatticeLink`] into a [`LatticeLinkCanonical`].
    ///
    /// See [`LatticeCyclic::link_canonical`] and [`LatticeLinkCanonical`].
    #[allow(clippy::wrong_self_convention)] // this is not the self which is converted but the LatticeLink
    #[must_use]
    #[inline]
    pub fn into_canonical(&self, l: LatticeLink<D>) -> LatticeLinkCanonical<D> {
        self.link_canonical(*l.pos(), *l.dir())
    }

    /// Get the number of points in a single direction.
    ///
    /// use [`LatticeCyclic::number_of_points`] for the total number of points.
    /// Not to confuse with [`LatticeCyclic::dim_st`] which is the dimension of space-time.
    #[must_use]
    #[inline]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Get an Iterator over all points of the lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// for i in [4, 8, 16, 32, 64].into_iter() {
    ///     let l = LatticeCyclic::<4>::new(1_f64, i)?;
    ///     assert_eq!(l.get_points().size_hint().0, l.number_of_points());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn get_points(&self) -> IteratorLatticePoint<'_, D> {
        IteratorLatticePoint::new(self)
    }

    /// Get an Iterator over all canonical link of the lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// for i in [4, 8, 16, 32, 64].into_iter() {
    ///     let l = LatticeCyclic::<4>::new(1_f64, i)?;
    ///     assert_eq!(
    ///         l.get_links().size_hint().0,
    ///         l.number_of_canonical_links_space()
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn get_links(&self) -> IteratorLatticeLinkCanonical<'_, D> {
        IteratorLatticeLinkCanonical::new(self)
    }

    /// create a new lattice with `size` the lattice size parameter, and `dim` the number of
    /// points in each spatial dimension.
    ///
    /// # Errors
    /// Size should be greater than 0 and dim greater or equal to 2, otherwise return an error.
    #[inline]
    pub fn new(size: Real, dim: usize) -> Result<Self, LatticeInitializationError> {
        if D == 0 {
            Err(LatticeInitializationError::ZeroDimension)
        } else if size <= 0_f64 || size.is_nan() || size.is_infinite() {
            Err(LatticeInitializationError::NonPositiveSize)
        } else if dim < 2 {
            Err(LatticeInitializationError::DimTooSmall)
        } else {
            Ok(Self { size, dim })
        }
    }

    /// Total number of canonical links oriented in space for a set time.
    ///
    /// Basically the number of element return by [`LatticeCyclic::get_links`].
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let l = LatticeCyclic::<4>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_canonical_links_space(), 8_usize.pow(4) * 4);
    ///
    /// let l = LatticeCyclic::<4>::new(1_f64, 16)?;
    /// assert_eq!(l.number_of_canonical_links_space(), 16_usize.pow(4) * 4);
    ///
    /// let l = LatticeCyclic::<3>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_canonical_links_space(), 8_usize.pow(3) * 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn number_of_canonical_links_space(&self) -> usize {
        self.number_of_points() * D
    }

    /// Total number of point in the lattice for a set time.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::LatticeCyclic;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let l = LatticeCyclic::<4>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_points(), 8_usize.pow(4));
    ///
    /// let l = LatticeCyclic::<4>::new(1_f64, 16)?;
    /// assert_eq!(l.number_of_points(), 16_usize.pow(4));
    ///
    /// let l = LatticeCyclic::<3>::new(1_f64, 8)?;
    /// assert_eq!(l.number_of_points(), 8_usize.pow(3));
    /// # Ok(())
    /// # }
    /// ```
    /// # Panics
    /// Panics if the dimensions D is bigger than [`u32::MAX`]
    #[must_use]
    #[inline]
    pub fn number_of_points(&self) -> usize {
        self.dim().pow(D.try_into().expect("conversion error"))
    }

    /// Return the lattice size factor.
    #[must_use]
    #[inline]
    pub const fn size(&self) -> Real {
        self.size
    }

    /// Get the next point in the lattice following the direction `dir`.
    /// It follows the Cyclic property of the lattice.
    ///
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclic, DirectionEnum, LatticePoint};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let point = LatticePoint::<4>::from([1, 0, 2, 0]);
    /// assert_eq!(
    ///     lattice.add_point_direction(point, &DirectionEnum::XPos.into()),
    ///     LatticePoint::from([2, 0, 2, 0])
    /// );
    /// // In the following case we get [_, 3, _, _] because `dim = 4`, and this lattice is Cyclic.
    /// assert_eq!(
    ///     lattice.add_point_direction(point, &DirectionEnum::YNeg.into()),
    ///     LatticePoint::from([1, 3, 2, 0])
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn add_point_direction(
        &self,
        point: LatticePoint<D>,
        dir: &Direction<D>,
    ) -> LatticePoint<D> {
        self.add_point_direction_n(point, dir, 1)
    }

    /// Returns the point given y moving `shift_number` times in direction `dir` from position `point`.
    /// It follows the Cyclic property of the lattice.
    ///
    /// It is equivalent of doing [`Self::add_point_direction`] n times.
    /// # Example
    /// ```
    /// # use lattice_qcd_rs::lattice::{LatticeCyclic, DirectionEnum, LatticePoint};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lattice = LatticeCyclic::<4>::new(1_f64, 4)?;
    /// let point = LatticePoint::<4>::from([1, 0, 2, 0]);
    /// assert_eq!(
    ///     lattice.add_point_direction_n(point, &DirectionEnum::XPos.into(), 2),
    ///     LatticePoint::from([3, 0, 2, 0])
    /// );
    /// // In the following case we get [_, 1, _, _] because `dim = 4`, and this lattice is Cyclic.
    /// assert_eq!(
    ///     lattice.add_point_direction_n(point, &DirectionEnum::YNeg.into(), 3),
    ///     LatticePoint::from([1, 1, 2, 0])
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn add_point_direction_n(
        &self,
        mut point: LatticePoint<D>,
        dir: &Direction<D>,
        shift_number: usize,
    ) -> LatticePoint<D> {
        let shift_number = shift_number % self.dim(); // we ensure that shift_number < % self.dim()
        if dir.is_positive() {
            point[dir.index()] = (point[dir.index()] + shift_number) % self.dim();
        } else {
            let dir_pos = dir.to_positive();
            if point[dir_pos.index()] < shift_number {
                point[dir_pos.index()] = self.dim() - (shift_number - point[dir_pos.index()]);
            } else {
                point[dir_pos.index()] = (point[dir_pos.index()] - shift_number) % self.dim();
            }
        }
        point
    }

    /// Returns whether the number of canonical link is the same as the length of `links`.
    #[must_use]
    #[inline]
    pub fn has_compatible_length_links(&self, links: &LinkMatrix) -> bool {
        self.number_of_canonical_links_space() == links.len()
    }

    /// Returns wether the number of point is the same as the length of `e_field`.
    #[must_use]
    #[inline]
    pub fn has_compatible_length_e_field(&self, e_field: &EField<D>) -> bool {
        self.number_of_points() == e_field.len()
    }

    /// Returns the length is compatible for both `links` and `e_field`.
    /// See [`Self::has_compatible_length_links`] and [`Self::has_compatible_length_e_field`].
    #[must_use]
    #[inline]
    pub fn has_compatible_length(&self, links: &LinkMatrix, e_field: &EField<D>) -> bool {
        self.has_compatible_length_links(links) && self.has_compatible_length_e_field(e_field)
    }
}

impl<const D: usize> Display for LatticeCyclic<D> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cyclic lattice with {}^{} points and spacing {}",
            self.dim, D, self.size
        )
    }
}

#[cfg(test)]
mod test {
    use super::LatticeCyclic;

    #[test]
    fn lattice() {
        let lattice = LatticeCyclic::<3>::new(1_f64, 8).expect("lattice has an error");
        assert_eq!(
            lattice.to_string(),
            "Cyclic lattice with 8^3 points and spacing 1"
        );
    }
}
