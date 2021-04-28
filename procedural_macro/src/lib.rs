//! Procedural macro for lattice_qcd_rs
//!
//! For internal use only. Mainly it has macro for implementation of
//! `lattice_qcd_rs::lattice::Direction`

#![allow(clippy::needless_return)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::cognitive_complexity)]
//#![warn(clippy::default_numeric_fallback)]
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
//#![warn(clippy::semicolon_if_nothing_returned)]
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

extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use proc_macro::TokenStream;
use quote::quote;

const MAX_DIM: usize = 127;

/// Implement `DirectionList` for `Direction` of `U1` to `U127`.
/// Using const generics might render this unecessary.
#[proc_macro]
pub fn implement_direction_list(_item: TokenStream) -> TokenStream {
    let mut implem = vec![];
    for i in 1_usize..=MAX_DIM {
        let mut array_direction = vec![];
        let mut array_direction_positives = vec![];
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
        let s = quote! {
            const #u_dir_ident: [Direction<#i>; #i * 2] = [ #(#array_direction),* ];
            const #u_dir_pos_ident: [Direction<#i>; #i] = [ #(#array_direction_positives),* ];
            impl DirectionList for Direction<#i> {
                #[inline]
                fn get_all_directions() -> & 'static [Self] {
                    &#u_dir_ident
                }
                #[inline]
                fn get_all_positive_directions() -> & 'static [Self] {
                    &#u_dir_pos_ident
                }
            }
        };
        implem.push(s);
    }
    let final_stream = quote! {
        #(#implem)*
    };
    final_stream.into()
}

const MAX_DIM_FROM_IMPLEM: usize = 10;

/// Implement trait [`From`] and [`std::convert::TryFrom`] and for directions
#[proc_macro]
pub fn implement_direction_from(_item: TokenStream) -> TokenStream {
    // implementation of the error returned by the TryFrom trait.
    let mut implem = vec![quote! {
        use std::convert::TryFrom;

        /// Error return by try from for Directions
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub enum ErrorDirectionConversion {
            /// the index is out of bound
            IndexOutOfBound,
        }

        impl core::fmt::Display for ErrorDirectionConversion{
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "the index is out of bound")
            }
        }

        impl std::error::Error for ErrorDirectionConversion {}
    }];

    for i in 1_usize..MAX_DIM_FROM_IMPLEM {
        for j in i + 1..=MAX_DIM_FROM_IMPLEM {
            //let u_ident_from = syn::Ident::new(&format!("U{}", i), proc_macro2::Span::call_site());
            //let u_ident_to = syn::Ident::new(&format!("U{}", j), proc_macro2::Span::call_site());
            implem.push(quote! {
                impl From<Direction<#i>> for Direction<#j> {
                    fn from(from: Direction<#i>) -> Self {
                        Self::new(from.index(), from.is_positive()).unwrap()
                    }
                }

                impl From<&Direction<#i>> for Direction<#j> {
                    fn from(from: &Direction<#i>) -> Self {
                        Self::new(from.index(), from.is_positive()).unwrap()
                    }
                }

                impl TryFrom<Direction<#j>> for Direction<#i> {
                    type Error = ErrorDirectionConversion;
                    fn try_from(from: Direction<#j>) -> Result<Self, Self::Error> {
                        Self::new(from.index(), from.is_positive())
                            .ok_or(ErrorDirectionConversion::IndexOutOfBound)
                    }
                }

                impl TryFrom<&Direction<#j>> for Direction<#i> {
                    type Error = ErrorDirectionConversion;
                    fn try_from(from: &Direction<#j>) -> Result<Self, Self::Error> {
                        Self::new(from.index(), from.is_positive())
                            .ok_or(ErrorDirectionConversion::IndexOutOfBound)
                    }
                }
            });
        }
    }

    let final_stream = quote! {
        #(#implem)*
    };
    final_stream.into()
}
