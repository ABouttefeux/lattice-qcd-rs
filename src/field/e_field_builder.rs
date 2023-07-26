use std::borrow::Cow;
use std::fmt::Debug;
use std::num::NonZeroUsize;

use rand::distributions::Distribution;
use rand_distr::Normal;
#[cfg(feature = "serde-serialize")]
use serde::Serialize;

use super::EField;
use crate::builder::GenType;
use crate::lattice::LatticeCyclic;
use crate::{CMatrix3, Real};

#[derive(Debug, Clone)]
enum DistributionForBuilder<'dis, Dis: Distribution<Real> + ?Sized> {
    Default(Normal<Real>),
    Given(&'dis Dis), // TODO box dyn
}

impl<'d, Dis: Distribution<Real> + ?Sized> Default for DistributionForBuilder<'d, Dis> {
    fn default() -> Self {
        Self::Default(Normal::new(0_f64, 0.5_f64).unwrap())
    }
}

#[derive(Clone, Debug)]
pub struct EFieldProceduralBuilder<
    'rng,
    'lat,
    'dis,
    Rng: rand::Rng + ?Sized,
    Dis: Distribution<Real> + ToOwned + ?Sized,
    const D: usize,
> {
    gen_type: GenType<'rng, Rng>,
    lattice: Cow<'lat, LatticeCyclic<D>>,
    distribution: DistributionForBuilder<'dis, Dis>,
}

/*
impl<'r, 'l, 'd, Rng: rand::Rng + ?Sized, Dis, const D: usize> Debug
    for EFieldProceduralBuilder<'r, 'l, 'd, Rng, Dis, D>
where
    Dis: Distribution<Real> + ?Sized + ToOwned + Debug,
    Rng: Debug,
    Dis::Owned: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EFieldProceduralBuilder")
            .field("gen_type", &self.gen_type)
            .field("lattice", &self.lattice)
            .field("distribution", &self.distribution)
            .finish()
    }
}
*/

impl<
        'rng,
        'lat,
        'dis,
        Rng: rand::Rng + ?Sized,
        Dis: Distribution<Real> + ToOwned + ?Sized,
        const D: usize,
    > EFieldProceduralBuilder<'rng, 'lat, 'dis, Rng, Dis, D>
{
    pub fn new(lattice: impl Into<Cow<'lat, LatticeCyclic<D>>>) -> Self {
        Self {
            gen_type: GenType::Cold,
            lattice: lattice.into(),
            distribution: DistributionForBuilder::default(),
        }
    }
}
