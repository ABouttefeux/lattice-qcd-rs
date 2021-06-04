use std::num::NonZeroUsize;

#[cfg(feature = "serde-serialize")]
use serde::Serialize;

use super::LinkMatrix;
use crate::builder::GenType;
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

/// Consuming [`LinkMatrix`] builder.
/// There is two way to startt the builder, [`LinkMatrixBuilder::new_from_data`]
/// [`LinkMatrixBuilder::new_procedural`].
///
/// The first one juste move the data given in the
/// [`LinkMatrix`] and does not require another configuration.
///
/// The seconde one will build a [`LinkMatrix`] procedurally and accept three configurations.
/// [`LinkMatrixBuilder::set_cold`] that generate a configuration with only indentity matrices.
/// [`LinkMatrixBuilder::set_hot_deterministe`] choose randomly every link matrices with a SU(3)
/// matrix unfiformly distributed in a reproductible way
/// [`LinkMatrixBuilder::set_hot_threaded`] also chooses random matrices as above but does it
/// with multiple thread and is not deterministe.
#[derive(Debug, PartialEq, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
pub struct LinkMatrixBuilder<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize> {
    builder_type: LinkMatrixBuilderType<'rng, 'lat, Rng, D>,
}

impl<'rng, 'lat, Rng: rand::Rng + ?Sized, const D: usize> LinkMatrixBuilder<'rng, 'lat, Rng, D> {
    /// This method take an array of data as the base contruction of [`LinkMatrix`].
    ///
    /// Using this methode has no other configuration and can be direcly build.
    /// It is equivalent to [`LinkMatrix::new`]
    /// # Example
    /// ```
    /// use lattice_qcd_rs::field::LinkMatrixBuilder;
    /// use lattice_qcd_rs::CMatrix3;
    /// use rand::rngs::ThreadRng;
    ///
    /// let vec = vec![CMatrix3::identity(); 16];
    /// let links = LinkMatrixBuilder::<'_, '_, ThreadRng, 4>::new_from_data(vec.clone()).build();
    /// assert_eq!(&vec, links.as_vec());
    /// ```
    pub fn new_from_data(data: Vec<CMatrix3>) -> Self {
        Self {
            builder_type: LinkMatrixBuilderType::Data(data),
        }
    }

    /// Initialize the builder to use procedural generation.
    ///
    /// By default the generation type is set to cold.
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::field::LinkMatrixBuilder;
    /// use lattice_qcd_rs::lattice::LatticeCyclique;
    /// use lattice_qcd_rs::CMatrix3;
    /// use rand::rngs::ThreadRng;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lat = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrixBuilder::<'_, '_, ThreadRng, 3>::new_procedural(&lat).build();
    /// assert!(lat.has_compatible_lenght_links(&links));
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_procedural(l: &'lat LatticeCyclique<D>) -> Self {
        Self {
            builder_type: LinkMatrixBuilderType::Generated(l, GenType::Cold),
        }
    }

    /// Change the methode to a cold generation, i.e. all links are set to the identity.
    /// Does not affect generactor build with [`LinkMatrixBuilder::new_from_data`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::field::LinkMatrixBuilder;
    /// use lattice_qcd_rs::lattice::LatticeCyclique;
    /// use lattice_qcd_rs::CMatrix3;
    /// use rand::rngs::ThreadRng;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lat = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrixBuilder::<'_, '_, ThreadRng, 3>::new_procedural(&lat)
    ///     .set_cold()
    ///     .build();
    /// for m in &links {
    ///     assert_eq!(m, &CMatrix3::identity());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_cold(mut self) -> Self {
        match self.builder_type {
            LinkMatrixBuilderType::Data(_) => {}
            LinkMatrixBuilderType::Generated(l, _) => {
                self.builder_type = LinkMatrixBuilderType::Generated(l, GenType::Cold);
            }
        }
        self
    }

    /// Change the methode to a hot determinist generation, i.e. all links generated randomly in a single thread.
    /// Does not affect generactor build with [`LinkMatrixBuilder::new_from_data`].
    ///
    /// # Example
    /// ```
    /// use lattice_qcd_rs::field::LinkMatrixBuilder;
    /// use lattice_qcd_rs::lattice::LatticeCyclique;
    /// use lattice_qcd_rs::CMatrix3;
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let mut rng = StdRng::seed_from_u64(0); // change the seed
    /// let lat = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let links = LinkMatrixBuilder::<'_, '_, StdRng, 3>::new_procedural(&lat)
    ///     .set_hot_deterministe(&mut rng)
    ///     .build();
    /// let mut rng_2 = StdRng::seed_from_u64(0); // same seed as before
    /// let links_2 = LinkMatrixBuilder::<'_, '_, StdRng, 3>::new_procedural(&lat)
    ///     .set_hot_deterministe(&mut rng_2)
    ///     .build();
    /// assert_eq!(links, links_2);
    /// # Ok(())
    /// # }
    /// ```
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

    /// Change the methode to a hot determinist generation, i.e. all links generated randomly using multiple threads.
    /// Does not affect generactor build with [`LinkMatrixBuilder::new_from_data`].
    ///
    /// # Example
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use lattice_qcd_rs::error::ImplementationError;
    /// use lattice_qcd_rs::field::LinkMatrixBuilder;
    /// use lattice_qcd_rs::lattice::LatticeCyclique;
    /// use lattice_qcd_rs::CMatrix3;
    /// use rand::rngs::ThreadRng;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let lat = LatticeCyclique::<3>::new(1_f64, 4)?;
    /// let number_of_threads =
    ///     NonZeroUsize::new(4).ok_or(ImplementationError::OptionWithUnexpectedNone)?;
    /// let links = LinkMatrixBuilder::<'_, '_, ThreadRng, 3>::new_procedural(&lat)
    ///     .set_hot_threaded(number_of_threads)
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
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

    /// Therminal methode to build the [`LinkMatrix`]
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
        let m = LinkMatrixBuilder::<'_, '_, rand::rngs::ThreadRng, 3>::new_procedural(&lattice)
            .set_cold()
            .build();
        assert_eq!(m, LinkMatrix::new_cold(&lattice));

        let mut rng = StdRng::seed_from_u64(SEED_RNG);
        let builder = LinkMatrixBuilder::<'_, '_, _, 3>::new_procedural(&lattice)
            .set_hot_deterministe(&mut rng);
        let m = builder.clone().build();
        assert_eq!(m, builder.build());
        let _ = LinkMatrixBuilder::<'_, '_, rand::rngs::ThreadRng, 3>::new_procedural(&lattice)
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
