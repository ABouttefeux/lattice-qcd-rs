#![doc = include_str!("../README.md")]
//
#![allow(clippy::needless_return)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::implicit_hasher)]
#![warn(clippy::implicit_saturating_sub)]
#![warn(clippy::imprecise_flops)]
#![warn(clippy::large_types_passed_by_value)]
#![warn(clippy::macro_use_imports)]
#![warn(clippy::manual_ok_or)]
#![warn(clippy::missing_const_for_fn)]
#![warn(clippy::needless_pass_by_value)]
#![warn(clippy::non_ascii_literal)]
#![warn(clippy::semicolon_if_nothing_returned)]
#![warn(clippy::suboptimal_flops)]
#![warn(clippy::todo)]
#![warn(clippy::trivially_copy_pass_by_ref)]
#![warn(clippy::type_repetition_in_bounds)]
#![warn(clippy::unreadable_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::unused_self)]
#![warn(clippy::unnecessary_wraps)]
#![warn(clippy::missing_errors_doc)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![doc(html_root_url = "https://docs.rs/lattice_qcd_rs-procedural_macro/0.2.0")]

/// Only contains the version test.
#[cfg(test)]
mod test;

use proc_macro::TokenStream;
use quote::quote;

/// Maximum dimension to impl [`Direction`] for.
const MAX_DIM: usize = 127;

/// Implement [`DirectionList`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/trait.DirectionList.html)
/// for [`Direction`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/struct.Direction.html) of `1` to `127` the value of `MAX_DIM`.
///
/// Using const generics might render this unnecessary. Waiting for stabilization of feature(generic_const_exprs).
#[proc_macro]
pub fn implement_direction_list(_item: TokenStream) -> TokenStream {
    let mut implem = Vec::with_capacity(MAX_DIM);

    // creation of the vec containing the directions.
    for i in 1_usize..=MAX_DIM {
        let mut array_direction = Vec::with_capacity(MAX_DIM);
        let mut array_direction_positives = Vec::with_capacity(MAX_DIM);
        for j in 0..i {
            array_direction.push(quote! {
                Direction{index_dir: #j, is_positive: true},
                Direction{index_dir: #j, is_positive: false}
            });
            array_direction_positives.push(quote! {
                Direction{index_dir: #j, is_positive: true}
            });
        }
        //let u_ident = syn::Ident::new(&format!("U{}", i), proc_macro2::Span::call_site());
        let u_dir_ident = syn::Ident::new(&format!("U{}_DIR", i), proc_macro2::Span::call_site());
        let u_dir_pos_ident =
            syn::Ident::new(&format!("U{}_DIR_POS", i), proc_macro2::Span::call_site());
        // we store the values in array so we can access them as fast as possible.
        let s = quote! {
            const #u_dir_ident: [Direction<#i>; #i * 2] = [ #(#array_direction),* ];
            const #u_dir_pos_ident: [Direction<#i>; #i] = [ #(#array_direction_positives),* ];
            impl DirectionList for Direction<#i> {
                #[inline]
                fn directions() -> & 'static [Self] {
                    &#u_dir_ident
                }
                #[inline]
                fn positive_directions() -> & 'static [Self] {
                    &#u_dir_pos_ident
                }
            }
        };
        implem.push(s);
    }
    // We need to concat the final implems togethers.
    let final_stream = quote! {
        #(#implem)*
    };
    final_stream.into()
}

/// The max dimension to imply the [`From`] and [`std::convert::TryFrom`] and for
/// [`Direction`](https://abouttefeux.github.io/lattice-qcd-rs/lattice_qcd_rs/lattice/struct.Direction.html).
///
/// It takes just way to long for 127.
const MAX_DIM_FROM_IMPLEM: usize = 10;

/// Implement trait [`From`] and [`std::convert::TryFrom`] and for directions
#[proc_macro]
pub fn implement_direction_from(_item: TokenStream) -> TokenStream {
    // implementation of the error returned by the TryFrom trait.
    let mut implem = vec![quote! {
        use std::convert::TryFrom;

        /// Error return by [`TryFrom`] for Directions.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
        #[non_exhaustive]
        pub enum DirectionConversionError {
            /// The index is out of bound, i.e. the direction axis does not exist in the lower space dimension.
            IndexOutOfBound,
        }

        impl std::fmt::Display for DirectionConversionError{
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::IndexOutOfBound => write!(f, "the index is out of bound, the direction axis does not exist in the lower space dimension"),
                }
            }
        }

        impl std::error::Error for DirectionConversionError {}
    }];

    for i in 1_usize..MAX_DIM_FROM_IMPLEM {
        for j in i + 1..=MAX_DIM_FROM_IMPLEM {
            //let u_ident_from = syn::Ident::new(&format!("U{}", i), proc_macro2::Span::call_site());
            //let u_ident_to = syn::Ident::new(&format!("U{}", j), proc_macro2::Span::call_site());
            implem.push(quote! {
                impl From<Direction<#i>> for Direction<#j> {
                    fn from(from: Direction<#i>) -> Self {
                        // i > j so so it is always Some.
                        Self::new(from.index(), from.is_positive()).unwrap()
                    }
                }

                impl From<&Direction<#i>> for Direction<#j> {
                    fn from(from: &Direction<#i>) -> Self {
                        // i > j so so it is always Some.
                        Self::new(from.index(), from.is_positive()).unwrap()
                    }
                }

                impl TryFrom<Direction<#j>> for Direction<#i> {
                    type Error = DirectionConversionError;
                    fn try_from(from: Direction<#j>) -> Result<Self, Self::Error> {
                        Self::new(from.index(), from.is_positive())
                            .ok_or(DirectionConversionError::IndexOutOfBound)
                    }
                }

                impl TryFrom<&Direction<#j>> for Direction<#i> {
                    type Error = DirectionConversionError;
                    fn try_from(from: &Direction<#j>) -> Result<Self, Self::Error> {
                        Self::new(from.index(), from.is_positive())
                            .ok_or(DirectionConversionError::IndexOutOfBound)
                    }
                }
            });
        }
    }

    let final_stream = quote! {
        #(#implem)*
    };
    // We need to concat the final implems togethers.
    final_stream.into()
}
