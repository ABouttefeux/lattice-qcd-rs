use std::num::NonZeroUsize;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::LinkMatrix;
use crate::lattice::LatticeCyclique;
use crate::CMatrix3;

#[non_exhaustive]
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
enum LinkMatrixBuilderType<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize> {
    /// Generate data procedurally
    Generated(&'lat LatticeCyclique<D>, GenType<'rng, Rng>),
    /// Data already existing
    Data(Vec<CMatrix3>),
}

/// Type of generation
#[non_exhaustive]
#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
enum GenType<'rng, Rng: rand::Rng + ?Sized> {
    /// Cold generation all ellements are set to the default
    Cold,
    /// Random deterministe
    #[cfg_attr(feature = "serde-serialize", serde(skip_deserializing))]
    HotDeterministe(&'rng mut Rng),
    /// Random deterministe but own the RNG (for instance the result of `clone`)
    HotDeterministeOwned(Box<Rng>),
    /// Random threaded (non deterministe)
    HotThreaded(NonZeroUsize),
}

impl<'rng, Rng: rand::Rng + Clone + ?Sized> Clone for GenType<'rng, Rng> {
    fn clone(&self) -> Self {
        match self {
            Self::Cold => Self::Cold,
            Self::HotDeterministe(rng_ref) => {
                Self::HotDeterministeOwned(Box::new((*rng_ref).clone()))
            }
            Self::HotDeterministeOwned(rng_box) => Self::HotDeterministeOwned(rng_box.clone()),
            Self::HotThreaded(n) => Self::HotThreaded(*n),
        }
    }
}

impl<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize>
    LinkMatrixBuilderType<'rng, 'lat, Rng, D>
{
    pub fn into_link_matrix(self) -> LinkMatrix {
        match self {
            Self::Data(data) => LinkMatrix::new(data),
            Self::Generated(l, gen_type) => match gen_type {
                GenType::Cold => LinkMatrix::new_cold(l),
                GenType::HotDeterministe(rng) => LinkMatrix::new_deterministe(l, rng),
                // the unwrap is safe because n is non zero
                // there is a possibility to panic in a thread but very unlikly
                // (either something break in this API or in thread_rng())
                GenType::HotDeterministeOwned(mut rng_box) => {
                    LinkMatrix::new_deterministe(l, &mut rng_box)
                }
                GenType::HotThreaded(n) => LinkMatrix::new_random_threaded(l, n.get()).unwrap(),
            },
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
pub struct LinkMatrixBuilder<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize> {
    builder_type: LinkMatrixBuilderType<'rng, 'lat, Rng, D>,
}

impl<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize> LinkMatrixBuilder<'rng, 'lat, Rng, D> {
    pub fn new_from_data(data: Vec<CMatrix3>) -> Self {
        Self {
            builder_type: LinkMatrixBuilderType::Data(data),
        }
    }

    pub fn new_generated(l: &'lat LatticeCyclique<D>) -> Self {
        Self {
            builder_type: LinkMatrixBuilderType::Generated(l, GenType::Cold),
        }
    }

    pub fn set_cold(mut self) -> Self {
        match self.builder_type {
            LinkMatrixBuilderType::Data(_) => {}
            LinkMatrixBuilderType::Generated(l, _) => {
                self.builder_type = LinkMatrixBuilderType::Generated(l, GenType::Cold);
            }
        }
        self
    }

    pub fn set_hot_deterministe(mut self, rng: &'rng mut Rng) -> Self {
        match self.builder_type {
            LinkMatrixBuilderType::Data(_) => {}
            LinkMatrixBuilderType::Generated(l, _) => {
                self.builder_type =
                    LinkMatrixBuilderType::Generated(l, GenType::HotDeterministe(rng));
            }
        }
        self
    }

    pub fn set_hot_threaded(mut self, number_of_threads: NonZeroUsize) -> Self {
        match self.builder_type {
            LinkMatrixBuilderType::Data(_) => {}
            LinkMatrixBuilderType::Generated(l, _) => {
                self.builder_type =
                    LinkMatrixBuilderType::Generated(l, GenType::HotThreaded(number_of_threads));
            }
        }
        self
    }

    pub fn build(self) -> LinkMatrix {
        self.builder_type.into_link_matrix()
    }
}

#[doc(hidden)]
impl<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize>
    From<LinkMatrixBuilderType<'rng, 'lat, Rng, D>> for LinkMatrixBuilder<'rng, 'lat, Rng, D>
{
    fn from(builder_type: LinkMatrixBuilderType<'rng, 'lat, Rng, D>) -> Self {
        Self { builder_type }
    }
}

impl<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize>
    From<LinkMatrixBuilder<'rng, 'lat, Rng, D>> for LinkMatrix
{
    fn from(builder: LinkMatrixBuilder<'rng, 'lat, Rng, D>) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod test {
    use std::num::NonZeroUsize;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::error::LatticeInitializationError;

    const SEED_RNG: u64 = 0x45_78_93_f4_4a_b0_67_f0;

    #[test]
    fn builder() -> Result<(), LatticeInitializationError> {
        let lattice = LatticeCyclique::<3>::new(1_f64, 10)?;
        let m = LinkMatrixBuilder::<'_, '_, rand::rngs::ThreadRng, 3>::new_generated(&lattice)
            .set_cold()
            .build();
        assert_eq!(m, LinkMatrix::new_cold(&lattice));

        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        let builder = LinkMatrixBuilder::<'_, '_, _, 3>::new_generated(&lattice)
            .set_hot_deterministe(&mut rng);
        let m = builder.clone().build();
        assert_eq!(m, builder.build());
        let _ = LinkMatrixBuilder::<'_, '_, rand::rngs::ThreadRng, 3>::new_generated(&lattice)
            .set_hot_threaded(NonZeroUsize::new(rayon::current_num_threads().min(1)).unwrap())
            .build();
        assert!(LinkMatrixBuilder::<'_, '_, _, 3>::new_from_data(vec![])
            .set_cold()
            .set_hot_deterministe(&mut rng)
            .set_hot_threaded(NonZeroUsize::new(1).unwrap())
            .build()
            .is_empty());
        assert_eq!(
            LinkMatrixBuilder::<'_, '_, rand::rngs::ThreadRng, 3>::new_from_data(
                vec![CMatrix3::identity(); 5]
            )
            .build()
            .as_ref(),
            vec![CMatrix3::identity(); 5]
        );
        assert_eq!(
            LinkMatrix::from(
                LinkMatrixBuilder::<'_, '_, rand::rngs::ThreadRng, 3>::new_from_data(
                    vec![CMatrix3::identity(); 100]
                )
            )
            .as_ref(),
            vec![CMatrix3::identity(); 100]
        );
        Ok(())
    }

    #[test]
    fn gen_type() {
        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        assert_eq!(
            GenType::<'_, StdRng>::Cold.clone(),
            GenType::<'_, StdRng>::Cold
        );
        assert_eq!(
            GenType::HotDeterministeOwned(Box::new(rng.clone())).clone(),
            GenType::HotDeterministeOwned(Box::new(rng.clone()))
        );
        assert_eq!(
            GenType::<'_, StdRng>::HotThreaded(NonZeroUsize::new(1).unwrap()).clone(),
            GenType::<'_, StdRng>::HotThreaded(NonZeroUsize::new(1).unwrap())
        );
        let gen_type = GenType::HotDeterministe(&mut rng);
        assert_ne!(gen_type.clone(), gen_type);
    }

    #[test]
    fn trait_misc() {
        let builder_type = LinkMatrixBuilderType::<'_, '_, StdRng, 10>::Data(vec![]);
        assert_eq!(
            LinkMatrixBuilder::from(builder_type.clone()).builder_type,
            builder_type
        );
    }
}
