
//! Procedural macro for lattice_qcd_rs

#![allow(clippy::needless_return)]
//#![warn(clippy::as_conversions)]
#![warn(clippy::cast_lossless)]
//#![warn(clippy::cast_precision_loss)]
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
//#![warn(clippy::type_repetition_in_bounds)]
#![warn(clippy::unreadable_literal)]
#![warn(clippy::unseparated_literal_suffix)]
#![warn(clippy::unused_self)]

#![warn(clippy::missing_errors_doc)]
#![warn(missing_docs)]

extern crate proc_macro;
extern crate quote;
extern crate syn;
extern crate proc_macro2;

use quote::quote;
use proc_macro::TokenStream;

/// Implement `DirectionList` for `Direction` of `U1` to `U127`
#[proc_macro]
pub fn implement_direction_list(_item: TokenStream) -> TokenStream {
    let mut implem = vec![];
    for i in 1_usize..=127_usize {
        let mut array_direction = vec![];
        let mut array_direction_positives = vec![];
        for j in 0..i {
            array_direction.push(
                quote!{
                    Direction{index_dir: #j, is_positive: true, _phantom: PhantomData},
                    Direction{index_dir: #j, is_positive: false, _phantom: PhantomData}
                }
            );
            array_direction_positives.push(
                quote!{
                    Direction{index_dir: #j, is_positive: true, _phantom: PhantomData}
                }
            );
        }
        let u_ident = syn::Ident::new(&format!("U{}", i), proc_macro2::Span::call_site());
        let u_dir_ident = syn::Ident::new(&format!("U{}_DIR", i), proc_macro2::Span::call_site());
        let u_dir_pos_ident = syn::Ident::new(&format!("U{}_DIR_POS", i), proc_macro2::Span::call_site());
        let s = quote!{
            const #u_dir_ident: [Direction<#u_ident>; #i * 2] = [ #(#array_direction),* ];
            const #u_dir_pos_ident: [Direction<#u_ident>; #i] = [ #(#array_direction_positives),* ];
            impl DirectionList for Direction<#u_ident> {
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
    let final_stream = quote!{
        #(#implem)*
    };
    final_stream.into()
}
