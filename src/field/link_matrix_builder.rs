use std::num::NonZeroUsize;

#[cfg(feature = "serde-serialize")]
use serde::Serialize;

use super::LinkMatrix;
use crate::lattice::LatticeCyclique;
use crate::CMatrix3;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
enum LinkMatrixBuilderType<'a, 'lat, Rng: rand::Rng + ?Sized, const D: usize> {
    Generated(&'lat LatticeCyclique<D>, GenType<'a, Rng>),
    Data(Vec<CMatrix3>),
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
enum GenType<'a, Rng: rand::Rng + ?Sized> {
    Cold,
    Hot(&'a mut Rng),
    HotThreaded(NonZeroUsize),
}

impl<'a, 'lat, Rng: rand::Rng + ?Sized, const D: usize> LinkMatrixBuilderType<'a, 'lat, Rng, D> {
    pub fn into_link_matrix(self) -> LinkMatrix {
        match self {
            Self::Data(data) => LinkMatrix::new(data),
            Self::Generated(l, gen_type) => match gen_type {
                GenType::Cold => LinkMatrix::new_cold(l),
                GenType::Hot(rng) => LinkMatrix::new_deterministe(l, rng),
                // the unwrap is safe because n is non zero
                // there is a possibility to panic in a thread but very unlikly
                // (either something break in this API or in thread_rng())
                GenType::HotThreaded(n) => LinkMatrix::new_random_threaded(l, n.get()).unwrap(),
            },
        }
    }
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize))]
pub struct LinkMatrixBuilder<'a, 'lat, Rng: rand::Rng + ?Sized, const D: usize> {
    builder_type: LinkMatrixBuilderType<'a, 'lat, Rng, D>,
}

impl<'a, 'lat, Rng: rand::Rng + ?Sized, const D: usize> LinkMatrixBuilder<'a, 'lat, Rng, D> {
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

    pub fn set_cold(&mut self) -> &mut Self {
        match self.builder_type {
            LinkMatrixBuilderType::Data(_) => {}
            LinkMatrixBuilderType::Generated(l, _) => {
                self.builder_type = LinkMatrixBuilderType::Generated(l, GenType::Cold);
            }
        }
        self
    }

    pub fn set_hot(&mut self, rng: &'a mut Rng) -> &mut Self {
        match self.builder_type {
            LinkMatrixBuilderType::Data(_) => {}
            LinkMatrixBuilderType::Generated(l, _) => {
                self.builder_type = LinkMatrixBuilderType::Generated(l, GenType::Hot(rng));
            }
        }
        self
    }

    pub fn set_hot_threaded(&mut self, number_of_threads: NonZeroUsize) -> &mut Self {
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
